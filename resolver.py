"""
Address resolver — the single entry point for translating a freeform Danish
address into the structured identifiers every downstream data source needs.

One DAWA lookup yields every handle we need: postnr/vejnavn/husnr for
Tinglysningen, adresse-UUID for Boligsiden, adgangsadresse-UUID for BBR and
Datafordeler, coordinates for Plandata WFS, kommunekode for statistics, etc.

Callers pass a `ResolvedAddress` to data-source modules so no module has to
repeat the DAWA round-trip. Results are cached in memory with a short TTL;
address data itself is stable (DAWA publishes new addresses, it doesn't
mutate existing ones), so the TTL mainly protects against traffic spikes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

import requests
from cachetools import TTLCache

DAWA_AUTOCOMPLETE = "https://api.dataforsyningen.dk/autocomplete"
DAWA_ADRESSER = "https://api.dataforsyningen.dk/adresser"

# Matches a Danish house number: one or more digits, optional lowercase/uppercase
# letter suffix (e.g. "12", "12A", "103a").
_HUSNR_RE = re.compile(r"\b(\d{1,3}[A-Za-z]?)\b")
# Matches a 4-digit postal code (1000-9999).
_POSTNR_RE = re.compile(r"\b([1-9]\d{3})\b")

# One hour is generous — DAWA addresses don't change mid-session.
_RESOLVE_CACHE: TTLCache[str, "ResolvedAddress"] = TTLCache(maxsize=2048, ttl=3600)


@dataclass(frozen=True)
class ResolvedAddress:
    """Canonical handle for a Danish address across every data source we use."""
    query: str                 # original freeform input (for logging/debug)
    label: str                 # human-readable "Street N, 1234 City"
    postnr: str
    vejnavn: str
    husnr: str
    etage: str | None
    door: str | None
    adresse_uuid: str          # DAWA "adresse" UUID — also used by Boligsiden
    adgang_uuid: str           # DAWA "adgangsadresse" UUID — BBR, Datafordeler
    kommunekode: str
    matrikelnr: str            # e.g. "4hf" — used to find tingbog via matrikel
    ejerlavskode: str          # e.g. "1290159" — pairs with matrikelnr
    ejerlavsnavn: str
    lat: float
    lng: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResolveError(RuntimeError):
    """Raised when a query cannot be resolved to a concrete address."""


def _best_hit(hits: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the 'most useful' hit from a DAWA adresser response.

    Prefer addresses without etage/dør (the ground-level / building entry) so
    a query for a building hits its adgangsadresse-level data rather than an
    arbitrary flat within it.
    """
    if not hits:
        return None
    for h in hits:
        if not h.get("etage") and not h.get("dør"):
            return h
    return hits[0]


def resolve(query: str) -> ResolvedAddress:
    """Resolve a freeform address to a `ResolvedAddress`, cached.

    Raises `ResolveError` if the query cannot be matched to a concrete
    address (e.g. too vague, typo, non-existent street). We also reject
    queries where the resolved result's postnr or husnr digits don't line
    up with what the user typed — DAWA's fuzzy autocomplete happily
    returns a plausible Danish address even for nonsense input, which
    would otherwise surface completely unrelated tingbog data.
    """
    key = query.strip().lower()
    if not key:
        raise ResolveError(
            "Tom søgning — skriv en adresse (gade, husnummer og postnummer)."
        )
    # Require at least a house number; without it we can't distinguish
    # a specific property from the hundreds on a given street.
    user_husnr_match = _HUSNR_RE.search(query)
    user_postnr_match = _POSTNR_RE.search(query)
    if not user_husnr_match:
        raise ResolveError(
            "Manglende husnummer — skriv fx 'Borgergade 12, 6752 Glejbjerg'."
        )
    user_husnr_digits = re.match(r"\d+", user_husnr_match.group(1)).group(0)
    user_postnr = user_postnr_match.group(1) if user_postnr_match else None

    cached = _RESOLVE_CACHE.get(key)
    if cached is not None:
        # Preserve the caller's original query verbatim so downstream
        # logging / debugging is truthful about what was asked.
        if cached.query != query:
            from dataclasses import replace
            return replace(cached, query=query)
        return cached

    # Step 1: autocomplete for fuzzy tolerance, to derive structured components.
    r = requests.get(
        DAWA_AUTOCOMPLETE,
        params={
            "q": query,
            "caretpos": len(query),
            "type": "adresse",
            "per_side": 5,
            "side": 1,
            "fuzzy": "true",
            "supplerendebynavn": "true",
        },
        timeout=10,
    )
    r.raise_for_status()
    suggestions = r.json()

    chosen_id: str | None = None
    for s in suggestions:
        data = s.get("data") or {}
        # "adresse" type gives us the adresse-UUID directly in data['id']
        if data.get("id"):
            chosen_id = data["id"]
            break
        # Fall back to postnr/vejnavn/husnr fields if only adgangsadresse-level
        if data.get("postnr") and data.get("vejnavn") and data.get("husnr"):
            # Look up the full adresse record below.
            break

    if not chosen_id:
        # Step 2: fall back to /adresser lookup on postnr/vejnavn/husnr.
        if not suggestions:
            raise ResolveError(
                f"No DAWA match for {query!r} — try being more specific "
                f"(include house number and postal code)."
            )
        d = suggestions[0].get("data") or {}
        if not (d.get("postnr") and d.get("vejnavn") and d.get("husnr")):
            raise ResolveError(
                f"DAWA only returned partial matches for {query!r}. "
                f"Try a more specific query."
            )
        r = requests.get(
            DAWA_ADRESSER,
            params={"postnr": d["postnr"], "vejnavn": d["vejnavn"], "husnr": d["husnr"]},
            timeout=10,
        )
        r.raise_for_status()
        hits = r.json()
        hit = _best_hit(hits)
        if not hit:
            raise ResolveError(f"No address record for {query!r}")
        chosen_id = hit["id"]

    # Step 3: fetch the full adresse record for complete metadata.
    # If the UUID from autocomplete points at a historically-deleted address
    # (DAWA returns 404 for those), fall back to a fresh /adresser query on
    # the structured components we already have.
    r = requests.get(f"{DAWA_ADRESSER}/{chosen_id}", timeout=10)
    if r.status_code == 404:
        hint = {}
        for s in suggestions:
            d = s.get("data") or {}
            if d.get("postnr") and d.get("vejnavn") and d.get("husnr"):
                hint = d
                break
        if not hint:
            raise ResolveError(
                f"DAWA UUID {chosen_id} for {query!r} is not a current address "
                f"and no structured fallback is available."
            )
        r = requests.get(
            DAWA_ADRESSER,
            params={"postnr": hint["postnr"], "vejnavn": hint["vejnavn"], "husnr": hint["husnr"]},
            timeout=10,
        )
        r.raise_for_status()
        hits = r.json()
        hit = _best_hit(hits)
        if not hit:
            raise ResolveError(f"No current address record for {query!r}")
        r = requests.get(f"{DAWA_ADRESSER}/{hit['id']}", timeout=10)
    r.raise_for_status()
    a = r.json()

    # On a full adresse record, vejstykke/husnr/postnummer/kommune/adgangspunkt
    # all live inside the embedded adgangsadresse object.
    adgang = a.get("adgangsadresse") or {}
    vejnavn = (adgang.get("vejstykke") or {}).get("navn", "")
    husnr = adgang.get("husnr", "")
    postnr = (adgang.get("postnummer") or {}).get("nr", "")
    postnavn = (adgang.get("postnummer") or {}).get("navn", "")
    etage = a.get("etage")
    door = a.get("dør")
    koord = (adgang.get("adgangspunkt") or {}).get("koordinater") or [0.0, 0.0]
    ejerlav = adgang.get("ejerlav") or {}
    matrikelnr = adgang.get("matrikelnr", "") or ""
    ejerlavskode = str(ejerlav.get("kode", "") or "")
    ejerlavsnavn = ejerlav.get("navn", "") or ""

    label = f"{vejnavn} {husnr}"
    if etage:
        label += f", {etage}."
    if door:
        label += f" {door}"
    label += f", {postnr} {postnavn}"

    # Sanity-check: DAWA's fuzzy autocomplete is eager — "<script>" or
    # "Kvakkelørevej 9999, 9999 Blabla" both resolve to *some* Danish
    # address. Reject anything whose house number or postal code does not
    # line up with what the user actually typed.
    resolved_husnr_digits = re.match(r"\d+", husnr).group(0) if husnr else ""
    if resolved_husnr_digits != user_husnr_digits:
        raise ResolveError(
            f"Ingen adresse matcher '{query}'. DAWA's bedste gæt var "
            f"{label!r}, men husnummeret stemmer ikke — tjek for stavefejl."
        )
    if user_postnr and postnr and user_postnr != postnr:
        raise ResolveError(
            f"Ingen adresse matcher '{query}'. DAWA's bedste gæt var "
            f"{label!r}, men postnummeret stemmer ikke."
        )

    resolved = ResolvedAddress(
        query=query,
        label=label,
        postnr=postnr,
        vejnavn=vejnavn,
        husnr=husnr,
        etage=etage,
        door=door,
        adresse_uuid=a["id"],
        adgang_uuid=adgang.get("id", ""),
        kommunekode=(adgang.get("kommune") or {}).get("kode", ""),
        matrikelnr=matrikelnr,
        ejerlavskode=ejerlavskode,
        ejerlavsnavn=ejerlavsnavn,
        lat=koord[1],
        lng=koord[0],
    )
    _RESOLVE_CACHE[key] = resolved
    return resolved
