"""
nosy-neighbour web server.

Serves a map-based UI, a JSON REST API, and an MCP server at POST /mcp.
"""

import copy
import logging
import os
import subprocess
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from mcp.server.fastmcp import FastMCP
import uvicorn

from nosy_neighbour import TinglysningClient, get_loan_type_info
from boligsiden import get_sales_history
from resolver import resolve as resolve_address, ResolveError

DAWA_REVERSE_URL = "https://api.dataforsyningen.dk/adgangsadresser/reverse"

log = logging.getLogger(__name__)

_client = TinglysningClient()


def _git_version() -> str:
    """Return a short 'sha branch date' string for injection into the HTML.

    Best-effort — returns 'unknown' if git is unavailable or the directory is
    not a checkout (e.g. when running from a container or tarball).
    """
    try:
        def g(*args: str) -> str:
            return subprocess.check_output(
                ["git", *args], stderr=subprocess.DEVNULL, cwd=os.path.dirname(os.path.abspath(__file__))
            ).decode().strip()
        sha = g("rev-parse", "--short", "HEAD")
        branch = g("rev-parse", "--abbrev-ref", "HEAD")
        date = g("log", "-1", "--format=%cI")
        return f"{sha} {branch} {date}"
    except Exception:
        return "unknown"


# Captured at import time. The service is always restarted on deploy
# (see /opt/nosyneighbour/update.sh), so this is safe; if the process is
# ever hot-reloaded in future, this needs to move into /api/version.
_VERSION = _git_version()

with open("templates/index.html") as f:
    _index_html = f.read()

# Inject a version marker at the top so `Ctrl+U` + search for "version" reveals
# exactly which commit is serving the page. Purely for operator use.
_index_html = f"<!-- version: {_VERSION} -->\n{_index_html}"


def _annotate_loan_types(tingbog: dict) -> dict:
    # Tingbog can come from _tingbog_cache, so deep-copy before mutating to
    # avoid polluting the cached object with annotations whose format may
    # change across releases.
    tingbog = copy.deepcopy(tingbog)
    for h in tingbog.get("haeftelser") or []:
        rente = float(h.get("rente") or 0)
        if (h.get("fastvariabel") == "variabel"
                and h.get("haeftelsestype") in ("Realkreditpantebrev", "Afgiftspantebrev")
                and rente > 0):
            h["loan_type_info"] = get_loan_type_info(rente, alias=h.get("alias"))
    return tingbog


# ── MCP server ────────────────────────────────────────────────────────────────
mcp_server = FastMCP("nosy-neighbour", stateless_http=True, json_response=True)


@mcp_server.tool()
def lookup_property(address: str) -> dict:
    """Look up Danish property records from tinglysning.dk.

    Given a freeform Danish address, returns owners (ejere), official
    valuation (vurdering) with equity estimate, mortgages and liens
    (hæftelser) with loan-type estimation for variable-rate realkreditlån,
    and easements (servitutter).
    """
    try:
        resolved = resolve_address(address)
    except ResolveError as e:
        return {"error": str(e)}
    try:
        tingbog = _client.lookup_address(
            resolved.postnr,
            resolved.vejnavn,
            resolved.husnr,
            matrikelnr=resolved.matrikelnr or None,
            ejerlavskode=resolved.ejerlavskode or None,
        )
    except RuntimeError as e:
        return {"error": str(e)}
    if tingbog is None:
        return {"error": "No property data found"}
    return _annotate_loan_types(tingbog)


@mcp_server.tool()
def lookup_sales_history(address: str) -> dict:
    """Look up historical sale prices for a Danish address from Boligsiden.

    Given a freeform Danish address, returns every recorded sale of that
    exact address with date, price (DKK), area (m²), price per m², and
    sale type (normal=Almindeligt salg, family=Familiehandel,
    auction=Tvangsauktion). Sorted newest first.
    """
    try:
        resolved = resolve_address(address)
    except ResolveError as e:
        return {"error": str(e)}
    try:
        return get_sales_history(resolved)
    except requests.RequestException as e:
        return {"error": f"Boligsiden unreachable: {e}"}


# ── FastAPI app ───────────────────────────────────────────────────────────────
_mcp_asgi = mcp_server.streamable_http_app()  # lazily initialises session_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp_server.session_manager.run():
        yield


app = FastAPI(title="nosy-neighbour", lifespan=lifespan)

# Self-hosted Leaflet (and any future static assets). Mounted before the MCP
# catch-all on / so /static/* routes resolve here. StaticFiles sets
# Last-Modified + ETag automatically; Cloudflare caches .js/.css by default.
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/api/autocomplete")
def autocomplete(q: str = Query(...)):
    try:
        results = _client.autocomplete_address(q)
    except requests.RequestException as e:
        log.warning("autocomplete upstream error: %s", e)
        raise HTTPException(status_code=502, detail="DAWA unreachable")
    return [
        {
            "label": r["forslagstekst"],
            "postnr": d["postnr"],
            "vejnavn": d["vejnavn"],
            "husnr": d["husnr"],
            "lat": d["y"],
            "lng": d["x"],
        }
        for r in results
        if (d := r.get("data", {})) and d.get("postnr") and d.get("vejnavn") and d.get("husnr")
    ]


@app.get("/api/reverse")
def reverse(lat: float = Query(...), lng: float = Query(...)):
    # DAWA reverse silently ignores maks_afstand and defaults to EPSG:25832 —
    # pass srid=4326 explicitly and post-validate distance ourselves so ocean
    # and cross-border clicks don't silently resolve to a distant address.
    try:
        resp = requests.get(
            DAWA_REVERSE_URL,
            params={"x": lng, "y": lat, "srid": 4326, "maks_afstand": 500},
            timeout=10,
        )
    except requests.RequestException as e:
        log.warning("reverse upstream error: %s", e)
        raise HTTPException(status_code=502, detail="DAWA unreachable")
    if resp.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail="Ingen adresse inden for 500 m af det valgte punkt.",
        )
    resp.raise_for_status()
    d = resp.json()
    a_lat = d["adgangspunkt"]["koordinater"][1]
    a_lng = d["adgangspunkt"]["koordinater"][0]
    # Haversine distance in metres
    from math import radians, sin, cos, asin, sqrt
    dlat = radians(a_lat - lat)
    dlng = radians(a_lng - lng)
    h = sin(dlat / 2) ** 2 + cos(radians(lat)) * cos(radians(a_lat)) * sin(dlng / 2) ** 2
    dist_m = 2 * 6371000 * asin(sqrt(h))
    if dist_m > 500:
        raise HTTPException(
            status_code=404,
            detail="Ingen adresse inden for 500 m af det valgte punkt.",
        )
    return {
        "label": d["adressebetegnelse"],
        "postnr": d["postnummer"]["nr"],
        "vejnavn": d["vejstykke"]["navn"],
        "husnr": d["husnr"],
        "lat": a_lat,
        "lng": a_lng,
    }


@app.get("/api/lookup")
def lookup(q: str = Query(...)):
    try:
        resolved = resolve_address(q)
    except ResolveError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        tingbog = _client.lookup_address(
            resolved.postnr,
            resolved.vejnavn,
            resolved.husnr,
            matrikelnr=resolved.matrikelnr or None,
            ejerlavskode=resolved.ejerlavskode or None,
        )
    except requests.Timeout:
        log.warning("tinglysning timeout for %r", q)
        raise HTTPException(
            status_code=504,
            detail="Tinglysning.dk svarer ikke lige nu — prøv igen om lidt.",
        )
    except requests.RequestException as e:
        log.warning("tinglysning upstream error for %r: %s", q, e)
        raise HTTPException(status_code=502, detail="Tinglysning.dk unreachable")
    except RuntimeError as e:
        msg = str(e)
        if "No property found" in msg:
            msg = (
                "Adressen har ingen selvstændig tingbog. Det sker typisk for "
                "andelsboliger, lejeboliger og ejendomme uden selvstændig BFE. "
                "Prøv foreningens hovedadresse."
            )
        raise HTTPException(status_code=404, detail=msg)
    if tingbog is None:
        raise HTTPException(status_code=404, detail="No property data found")
    return _annotate_loan_types(tingbog)


@app.get("/api/sales-history")
def sales_history(q: str = Query(...)):
    """Historical sale prices for a given address, sourced from Boligsiden."""
    try:
        resolved = resolve_address(q)
    except ResolveError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        return get_sales_history(resolved)
    except requests.RequestException as e:
        log.warning("sales-history upstream error: %s", e)
        raise HTTPException(status_code=502, detail="Boligsiden unreachable")


@app.get("/api/resolve")
def resolve_endpoint(q: str = Query(...)):
    """Return the structured identifiers nosy-neighbour derives for an address.

    Useful for debugging (why does source X not find my address?) and as a
    shared primitive for any client that wants to call multiple data-source
    endpoints without re-paying the DAWA round-trip cost.
    """
    try:
        return resolve_address(q).to_dict()
    except ResolveError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except requests.RequestException as e:
        log.warning("resolver upstream error: %s", e)
        raise HTTPException(status_code=502, detail="DAWA unreachable")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=_index_html)


# Cheeky little globe with glasses — served inline so we don't have to add
# a binary asset to the repo. SVG is fine for favicons in all modern browsers.
_FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <circle cx="32" cy="32" r="28" fill="#4a90d9"/>
  <path d="M8 32h48M32 6c8 7 8 45 0 52M32 6c-8 7-8 45 0 52M12 18c6 4 34 4 40 0M12 46c6-4 34-4 40 0"
        fill="none" stroke="#2e7d32" stroke-width="2" opacity="0.55"/>
  <path d="M6 34c5-5 10-5 14 0M24 34c2-3 6-3 8 0M46 34c4-5 9-5 12 0"
        fill="#a8d8a8" opacity="0.7"/>
  <g transform="translate(0,4)">
    <circle cx="22" cy="30" r="8" fill="none" stroke="#1a1a1a" stroke-width="3"/>
    <circle cx="42" cy="30" r="8" fill="none" stroke="#1a1a1a" stroke-width="3"/>
    <path d="M30 30h4" stroke="#1a1a1a" stroke-width="3" stroke-linecap="round"/>
    <path d="M14 28l-5-2M50 28l5-2" stroke="#1a1a1a" stroke-width="3" stroke-linecap="round"/>
    <circle cx="22" cy="30" r="6" fill="#ffffff" opacity="0.85"/>
    <circle cx="42" cy="30" r="6" fill="#ffffff" opacity="0.85"/>
    <circle cx="23" cy="31" r="1.6" fill="#1a1a1a"/>
    <circle cx="43" cy="31" r="1.6" fill="#1a1a1a"/>
  </g>
</svg>"""


@app.get("/favicon.ico")
@app.get("/favicon.svg")
def favicon():
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")


# Tell crawlers to stay out — the site is geo-blocked to DK and lives behind
# a free-tier Cloudflare WAF; there's nothing useful for search engines here
# and indexing only invites scraping attempts.
_ROBOTS_TXT = "User-agent: *\nDisallow: /\n"


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    return _ROBOTS_TXT


@app.get("/api/version")
def version():
    return {"version": _VERSION}


# Mount MCP last so FastAPI routes take priority when matching paths.
# streamable_http_app() registers its handler at /mcp inside the sub-app;
# mounting the sub-app at / keeps the final endpoint at POST /mcp.
app.mount("/", _mcp_asgi)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_certfile=os.environ.get("SSL_CERTFILE"),
        ssl_keyfile=os.environ.get("SSL_KEYFILE"),
    )
