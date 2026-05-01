"""
nosy-neighbour: Look up Danish property records from tinglysning.dk.

Find out who owns a property, what it's worth, what mortgages are on it,
and what type of loans they have -- all from a freeform address string.

Estimates variable-rate loan types (F-kort, F1, F3, F5) by matching
against Nationalbanken rate statistics. Definitively identifies loan types
when an ISIN is provided (via ESMA FIRDS).
"""

import base64
import hashlib
import json
import re
import time
from datetime import datetime, timedelta

import requests
from cachetools import TTLCache

BASE_URL = "https://www.tinglysning.dk/tinglysning/unsecrest"
DAWA_URL = "https://dawa.aws.dk/autocomplete"
DST_API_URL = "https://api.statbank.dk/v1/data"
FIRDS_URL = "https://registers.esma.europa.eu/solr/esma_registers_firds/select"

# Maps DST RENTFIX codes to human-readable loan type names
RENTFIX_LOAN_TYPES = {
    "1M3M": "F-kort",
    "1A": "F1",
    "3A": "F3",
    "5A": "F5",
    "S10A": "Fastforrentet",
}
RENTFIX_CODES = list(RENTFIX_LOAN_TYPES.keys())

# Rate delta below which we consider two loan types ambiguous.
_CLOSE_MATCH_THRESHOLD = 0.4
# Rate delta above which we consider the estimation uncertain.
_UNCERTAIN_THRESHOLD = 1.0

# Cache DST rate lookups (monthly data, refreshed rarely) and tingbog results
# to avoid repeating slow upstream calls during loan-type annotation and
# back-to-back property lookups.
_dst_cache: TTLCache = TTLCache(maxsize=256, ttl=86400)     # 24h
_tingbog_cache: TTLCache = TTLCache(maxsize=1024, ttl=600)  # 10min

# Ticker patterns that identify loan types in ESMA FIRDS data.
# Realkredit Danmark: 1RD...1IT=F1, 1RD...2IT=F3, 1RD...RF=F5
# Nykredit/Totalkredit: 1NYK...IT=F1, F3NYK...=F3, F6NYK...=F5
# Nordea Kredit: ...IT1Y=F1, ...IT2Y=F3, ...RF=F5
_FIRDS_LOAN_TYPE_PATTERNS = [
    # Specific suffixes first (most reliable)
    # 2IT and IT2Y must come before the generic IT pattern
    (re.compile(r"2IT"), "F3"),
    (re.compile(r"IT2Y"), "F3"),
    (re.compile(r"1IT|IT$|DKKIT"), "F1"),
    (re.compile(r"IT1Y"), "F1"),
    # RF must come before F3/F5/F6 prefix patterns to avoid false matches
    # on digits in maturity years (e.g. "F36" in 1RD10F36APRF)
    (re.compile(r"RF"), "F5"),
    # Explicit loan type prefixes (Nykredit/Totalkredit: F3NYK, F6NYK, F5NYK)
    (re.compile(r"F3NYK"), "F3"),
    (re.compile(r"F[56]NYK"), "F5"),
    # Fixed-rate patterns
    (re.compile(r"EA\d|OA\d"), "Fastforrentet"),
]


def _solve_altcha(challenge_data: dict) -> str:
    """Solve the ALTCHA proof-of-work challenge and return a base64 token."""
    algorithm = challenge_data["algorithm"]
    challenge = challenge_data["challenge"]
    salt = challenge_data["salt"]
    signature = challenge_data["signature"]
    max_number = challenge_data["maxnumber"]

    if algorithm != "SHA-256":
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    start = time.time()
    for number in range(max_number + 1):
        computed = hashlib.sha256((salt + str(number)).encode()).hexdigest()
        if computed == challenge:
            took = int((time.time() - start) * 1000)
            solution = {
                "algorithm": algorithm,
                "challenge": challenge,
                "number": number,
                "salt": salt,
                "signature": signature,
                "took": took,
            }
            token_json = json.dumps(solution, separators=(",", ":"))
            return base64.b64encode(token_json.encode()).decode()

    raise RuntimeError("Failed to solve ALTCHA challenge")


def _fetch_dst_rates(months: list[str]) -> dict:
    """Fetch effective rates and bidrag from DST for given months.

    Returns {month: {rentfix_code: {"effective": float, "bidrag": float, "coupon": float}}}
    """
    key = tuple(months)
    hit = _dst_cache.get(key)
    if hit is not None:
        return hit
    resp = requests.post(DST_API_URL, json={
        "table": "DNRNURI",
        "format": "JSONSTAT",
        "lang": "da",
        "variables": [
            {"code": "DATA", "values": ["AL51EFFR", "AL51BIDS"]},
            {"code": "INDSEK", "values": ["1430"]},
            {"code": "VALUTA", "values": ["DKK"]},
            {"code": "LØBETID1", "values": ["ALLE"]},
            {"code": "RENTFIX", "values": RENTFIX_CODES},
            {"code": "LAANSTR", "values": ["ALLE"]},
            {"code": "Tid", "values": months},
        ],
    }, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    values = data["dataset"]["value"]

    n_rent = len(RENTFIX_CODES)
    n_tid = len(months)

    result = {}
    for t_idx, month in enumerate(months):
        result[month] = {}
        for r_idx, rcode in enumerate(RENTFIX_CODES):
            eff_idx = r_idx * n_tid + t_idx
            bid_idx = n_rent * n_tid + r_idx * n_tid + t_idx
            eff = values[eff_idx]
            bid = values[bid_idx]
            if eff is not None and bid is not None:
                result[month][rcode] = {
                    "effective": eff,
                    "bidrag": bid,
                    "coupon": round(eff - bid, 4),
                }
    _dst_cache[key] = result
    return result


def _recent_months(n: int = 6) -> list[str]:
    """Return the last n months as DST time codes (e.g. '2025M03').

    Starts from 2 months ago to avoid requesting unpublished data.
    """
    # Start 2 months back so we never request unpublished data
    now = datetime.now()
    year, month = now.year, now.month - 2
    if month <= 0:
        year -= 1
        month += 12
    months = []
    for _ in range(n):
        months.append(f"{year}M{month:02d}")
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    return list(reversed(months))


def _months_for_alias(alias: str, n: int = 6) -> list[str]:
    """Return n months centred on the registration date in an alias (DD.MM.YYYY-...).

    Falls back to _recent_months() if the date cannot be parsed.
    """
    try:
        date = datetime.strptime(alias.split("-")[0], "%d.%m.%Y")
    except (ValueError, IndexError):
        return _recent_months(n)
    year, month = date.year, date.month - n // 2
    while month <= 0:
        month += 12
        year -= 1
    months = []
    for _ in range(n):
        months.append(f"{year}M{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def estimate_loan_type(rate: float, rates_by_month: dict) -> list[tuple[str, float]]:
    """Find the loan types whose coupon rates best match the given rate.

    Returns a list of (loan_type_name, best_distance) sorted by distance,
    with one entry per loan type (keeping only the best month match per type).
    """
    best_per_type: dict[str, float] = {}

    for month, rates in rates_by_month.items():
        for rcode, rate_data in rates.items():
            distance = abs(rate_data["coupon"] - rate)
            name = RENTFIX_LOAN_TYPES[rcode]
            if name not in best_per_type or distance < best_per_type[name]:
                best_per_type[name] = distance

    return sorted(
        [(name, round(dist, 4)) for name, dist in best_per_type.items()],
        key=lambda x: x[1],
    )


def lookup_isin(isin: str) -> dict | None:
    """Look up a bond ISIN in ESMA FIRDS and return its details.

    Returns a dict with keys: isin, ticker, short_name, loan_type, maturity,
    coupon, trading_start, trading_end. Returns None if not found.
    """
    resp = requests.get(FIRDS_URL, params={
        "q": f"isin:{isin}",
        "wt": "json",
        "rows": 1,
    }, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data["response"]["numFound"] == 0:
        return None

    doc = data["response"]["docs"][0]
    ticker = doc.get("gnr_full_name", "")
    short_name = doc.get("gnr_short_name", "")

    # Determine loan type from ticker
    loan_type = _classify_ticker(ticker)

    return {
        "isin": doc.get("isin"),
        "ticker": ticker,
        "short_name": short_name,
        "loan_type": loan_type,
        "maturity": doc.get("bnd_maturity_date", "")[:10],
        "coupon": doc.get("bnd_fixed_rate"),
        "trading_start": doc.get("mrkt_trdng_start_date", "")[:10],
        "trading_end": doc.get("mrkt_trdng_trmination_date", "")[:10],
    }


def _classify_ticker(ticker: str) -> str | None:
    """Extract loan type from a FIRDS bond ticker.

    Known patterns:
      Realkredit Danmark: 1RD...1IT → F1, 1RD...2IT → F3, 1RD...RF → F5
      Nykredit/Totalkredit: 1NYK...IT → F1, F3NYK... → F3, F6NYK... → F5
      Nordea Kredit: ...IT1Y → F1, ...IT2Y → F3, ...RF → F5
    """
    for pattern, loan_type in _FIRDS_LOAN_TYPE_PATTERNS:
        if pattern.search(ticker):
            return loan_type
    return None


def get_loan_type_info(rate: float, isin: str | None = None, alias: str | None = None, num_months: int = 6) -> dict:
    """Return structured loan type info for a variable-rate mortgage.

    Uses ISIN lookup via ESMA FIRDS when available (definitive), otherwise
    estimates by matching rate against DST Nationalbanken statistics.

    Return dict keys:
      source: "isin" | "estimated" | "unknown"
      loan_type: str | None
      (isin source) ticker, short_name, maturity, coupon
      (estimated source) uncertain: bool, close_to: list[str],
                         candidates: list[{name, delta}]
    """
    if isin:
        info = lookup_isin(isin)
        if info and info["loan_type"]:
            return {
                "source": "isin",
                "loan_type": info["loan_type"],
                "isin": isin,
                "ticker": info["ticker"],
                "short_name": info["short_name"],
                "maturity": info["maturity"],
                "coupon": info["coupon"],
            }

    try:
        months = _months_for_alias(alias, num_months) if alias else _recent_months(num_months)
        dst_rates = _fetch_dst_rates(months)
    except Exception:
        return {"source": "unknown"}

    candidates = estimate_loan_type(rate, dst_rates)
    if not candidates:
        return {"source": "unknown"}

    best_name, best_delta = candidates[0]
    close = [n for n, d in candidates[1:] if d - best_delta < _CLOSE_MATCH_THRESHOLD and d < _UNCERTAIN_THRESHOLD]
    return {
        "source": "estimated",
        "loan_type": None if best_delta > _UNCERTAIN_THRESHOLD else best_name,
        "uncertain": best_delta > _UNCERTAIN_THRESHOLD,
        "close_to": close,
        "candidates": [{"name": n, "delta": d} for n, d in candidates],
    }


class TinglysningClient:
    # Upstream tinglysning.dk has been observed to hang for minutes on
    # individual requests. Cap every call so one slow property doesn't tie
    # up a worker forever, but keep the read budget generous (45 s) so we
    # don't falsely give up on a property that's just slow.
    _TIMEOUT = (10, 45)  # (connect, read) seconds

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.tinglysning.dk/tmv/forespoergul",
        })
        self._token: str | None = None

    def _get_token(self) -> str:
        """Fetch and solve an ALTCHA challenge; reuse within the same session."""
        if self._token is None:
            resp = self.session.get(f"{BASE_URL}/altcha/fetchChallenge", timeout=self._TIMEOUT)
            resp.raise_for_status()
            self._token = _solve_altcha(resp.json())
        return self._token

    def autocomplete_address(self, query: str) -> list[dict]:
        """Autocomplete a Danish address using DAWA.

        Returns a list of suggestions with 'tekst' (text) and 'data' fields.
        """
        resp = self.session.get(DAWA_URL, params={
            "q": query,
            "caretpos": len(query),
            "type": "adgangsadresse",
            "per_side": 20,
            "side": 1,
            "fuzzy": "true",
            "supplerendebynavn": "true",
        }, timeout=self._TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def resolve_address(self, query: str) -> tuple[str, str, str]:
        """Resolve a freeform address string into (postnummer, vejnavn, husnummer).

        Uses DAWA fuzzy autocomplete to find the best match, then extracts
        the structured address components from the result.
        """
        results = self.autocomplete_address(query)
        # We need a result that has full address data (not just a street name suggestion)
        for r in results:
            data = r.get("data", {})
            if data.get("postnr") and data.get("vejnavn") and data.get("husnr"):
                return data["postnr"], data["vejnavn"], data["husnr"]

        raise RuntimeError(f"Could not resolve address: {query!r}\n"
                           f"  DAWA returned {len(results)} suggestions but none had full address data.\n"
                           f"  Try being more specific (e.g. include house number and postal code).")

    def _get_json(self, url: str, params: dict) -> dict:
        """GET a tinglysning endpoint, retrying once on stale connection or expired token.

        Two failure modes are handled:
        - RemoteDisconnected: keep-alive connection was closed by the server; retry opens a fresh one.
        - Empty/non-JSON body: ALTCHA token expired; discard cached token and retry.
        """
        params = dict(params)
        last_status: int = 0
        last_text: str = ""
        for attempt in range(2):
            params["token"] = self._get_token()
            try:
                resp = self.session.get(url, params=params, timeout=self._TIMEOUT)
            except requests.exceptions.ConnectionError:
                # Stale keep-alive connection — retry will open a fresh one
                if attempt == 1:
                    raise
                continue
            except requests.exceptions.Timeout:
                # Read/connect timeout — retry once with a fresh connection
                if attempt == 1:
                    raise
                continue
            resp.raise_for_status()
            last_status = resp.status_code
            last_text = resp.text
            if resp.content and resp.text.strip():
                try:
                    return resp.json()
                except ValueError:
                    pass
            # Empty or non-JSON body — token is stale, discard and retry once
            self._token = None
        # Both attempts exhausted without a JSON payload.
        raise RuntimeError(
            f"Invalid response from tinglysning.dk after token refresh "
            f"(status {last_status}): {last_text[:200] or '(empty)'}"
        )

    def search_property(self, postnummer: str, vejnavn: str, husnummer: str) -> list[dict]:
        """Search for a property by address components.

        Returns a list of matching properties with 'uuid', 'adresse', and 'bog'
        fields. Returns an empty list when tinglysning.dk reports success but
        has no match (items can be null in that case).
        """
        data = self._get_json(f"{BASE_URL}/ejendomsoeg/soeg", {
            "postnummer": postnummer,
            "vejnavn": vejnavn,
            "husnummer": husnummer,
        })
        if data.get("statuskode") != 0:
            raise RuntimeError(f"Search failed: {data.get('statustekst')}")
        return data.get("items") or []

    def get_tingbog(self, uuid: str) -> dict:
        """Fetch the full tingbog (property register) for a property UUID.

        Returns property details including owners, mortgages, easements, and valuation.
        """
        hit = _tingbog_cache.get(uuid)
        if hit is not None:
            return hit
        data = self._get_json(f"{BASE_URL}/ejendomsoeg/henttingbog/{uuid}", {})
        if data.get("statuskode") != 0:
            raise RuntimeError(f"Lookup failed: {data.get('statustekst')}")
        _tingbog_cache[uuid] = data
        return data

    def lookup_address(
        self,
        postnummer: str,
        vejnavn: str,
        husnummer: str,
        *,
        matrikelnr: str | None = None,
        ejerlavskode: str | None = None,
    ) -> dict:
        """Full lookup: search for a property and return its tingbog data.

        Combines search_property() and get_tingbog() into a single call.

        If the exact-address search returns no result (which happens for
        andelsboliger / almennyttige ejendomme where the individual dwelling
        has no separate tingbog), we fall back to asking DAWA for every
        address sharing the same matrikel, then try each as a tinglysning
        search target until one resolves. That lets us surface the parent
        foundation/association's tingbog without scanning a whole street.

        The returned tingbog gets a `_matrikel_fallback` dict with details
        about why it was used, or None when the exact address matched.
        """
        items = self.search_property(postnummer, vejnavn, husnummer)
        if items:
            tingbog = self.get_tingbog(items[0]["uuid"])
            tingbog["_matrikel_fallback"] = None
            return tingbog

        # No exact match — try to find the owning tingbog via matrikel.
        if matrikelnr and ejerlavskode:
            fallback = self._find_tingbog_by_matrikel(
                matrikelnr, ejerlavskode,
            )
            if fallback is not None:
                tingbog, parent_adresse = fallback
                tingbog["_matrikel_fallback"] = {
                    "matrikelnr": matrikelnr,
                    "ejerlavskode": ejerlavskode,
                    "parent_adresse": parent_adresse,
                }
                return tingbog

        raise RuntimeError("No property found for the given address")

    def _find_tingbog_by_matrikel(
        self,
        matrikelnr: str,
        ejerlavskode: str,
    ) -> tuple[dict, str] | None:
        """Find the tingbog for a matrikel by asking DAWA for its addresses.

        DAWA knows every adgangsadresse tied to a given (ejerlav, matrikel)
        pair. We iterate those addresses, ask tinglysning.dk for each, and
        return the first hit whose tingbog actually covers our matrikel.
        Far more targeted than scanning every tingbog on the street.

        Returns (tingbog_data, parent_address_label) or None.
        """
        r = requests.get(
            "https://api.dataforsyningen.dk/adgangsadresser",
            params={
                "ejerlavkode": ejerlavskode,
                "matrikelnr": matrikelnr,
                "struktur": "mini",
            },
            timeout=10,
        )
        r.raise_for_status()
        candidates = r.json() or []
        for c in candidates:
            postnr = c.get("postnr")
            vejnavn = c.get("vejnavn")
            husnr = c.get("husnr")
            if not (postnr and vejnavn and husnr):
                continue
            try:
                items = self.search_property(postnr, vejnavn, husnr)
            except RuntimeError:
                continue
            if not items:
                continue
            try:
                tingbog = self.get_tingbog(items[0]["uuid"])
            except RuntimeError:
                continue
            for mat in tingbog.get("matrikler") or []:
                if (mat.get("matrikelnummer") == matrikelnr
                        and str(mat.get("landsejerlavkode")) == ejerlavskode):
                    return tingbog, c.get("betegnelse", "")
        return None

    def lookup(self, query: str) -> dict:
        """Look up a property by freeform address string.

        Resolves the address via DAWA, then fetches the tingbog.
        """
        postnummer, vejnavn, husnummer = self.resolve_address(query)
        return self.lookup_address(postnummer, vejnavn, husnummer)


def _print_loan_type_estimate(rate: float, isin: str | None, alias: str | None = None):
    """Print loan type info: definitive if ISIN is available, estimated otherwise."""
    if isin:
        info = lookup_isin(isin)
        if info and info["loan_type"]:
            print(f"      Loan type: {info['loan_type']} (via ISIN {isin})")
            print(f"        Bond:     {info['short_name']}")
            print(f"        Ticker:   {info['ticker']}")
            print(f"        Maturity: {info['maturity']}")
            if info['coupon'] is not None:
                print(f"        Coupon:   {info['coupon']}%")
            return
        elif info:
            print(f"      ISIN {isin} found but loan type could not be determined from ticker: {info['ticker']}")
        else:
            print(f"      ISIN {isin} not found in ESMA FIRDS")

    try:
        months = _months_for_alias(alias, 6) if alias else _recent_months(6)
        dst_rates = _fetch_dst_rates(months)
    except Exception as e:
        print(f"      (Could not fetch DST rate data: {e})")
        return

    candidates = estimate_loan_type(rate, dst_rates)
    if not candidates:
        return

    best_name, best_delta = candidates[0]
    close = [(n, d) for n, d in candidates[1:] if d - best_delta < _CLOSE_MATCH_THRESHOLD and d < _UNCERTAIN_THRESHOLD]
    if best_delta > _UNCERTAIN_THRESHOLD:
        print(f"      Estimated loan type: uncertain (rate doesn't match data around registration date)")
    elif not close:
        print(f"      Estimated loan type: {best_name}")
    else:
        print(f"      Estimated loan type: {best_name} (but close to {', '.join(n for n, _ in close)})")
    for rank, (name, delta) in enumerate(candidates):
        marker = " <-- best match" if rank == 0 else ""
        print(f"        {name:14s} rate delta: {delta:.4f}%{marker}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Look up Danish property records on tinglysning.dk")
    parser.add_argument("address", nargs="+",
                        help="Address to look up. Can be a freeform string like "
                             "'Stålhøjen 24, 8240 Risskov' or separate parts: postnummer vejnavn husnummer")
    parser.add_argument("--isin", action="append", default=[],
                        metavar="PRIORITY:ISIN",
                        help="ISIN for a specific mortgage, as priority:ISIN (e.g. 2:DK0004632486). "
                             "Can be specified multiple times.")
    args = parser.parse_args()

    # Parse --isin args into {priority: isin}
    isin_map: dict[str, str] = {}
    for entry in args.isin:
        if ":" in entry:
            prio, isin = entry.split(":", 1)
            isin_map[prio] = isin
        else:
            print(f"Warning: ignoring --isin {entry} (expected format priority:ISIN)")

    client = TinglysningClient()
    query = " ".join(args.address)
    postnummer, vejnavn, husnummer = client.resolve_address(query)
    print(f"Resolved: {vejnavn} {husnummer}, {postnummer}")
    result = client.lookup_address(postnummer, vejnavn, husnummer)

    print(f"\nAddress: {result['adresse']}")
    print(f"Property type: {result['ejendomstype']}")

    if result.get("matrikler"):
        print("\nMatrikler:")
        for m in result["matrikler"]:
            print(f"  {m['matrikelnummer']} - {m['landsejerlavnavn']}")

    if result.get("vurdering"):
        v = result["vurdering"]
        print(f"\nValuation ({v['vurderingsdato']}):")
        print(f"  Property value: {v['ejendomsvaerdi']:,} DKK")
        print(f"  Land value:     {v['grundvaerdi']:,} DKK")
        print(f"  Municipality:   {v['kommune']}")

    if result.get("ejere"):
        print("\nOwners:")
        for e in result["ejere"]:
            print(f"  {e['navn']} ({e['andel']})")

    variable_aliases = {
        h["alias"]
        for h in result.get("haeftelser", [])
        if h.get("fastvariabel") == "variabel"
        and h.get("haeftelsestype") in ("Realkreditpantebrev", "Afgiftspantebrev")
        and float(h.get("rente") or 0) > 0
    }

    if result.get("haeftelser"):
        print("\nMortgages/Liens:")
        for h in result["haeftelser"]:
            rate_str = h['rente']
            fixed_var = 'fast' if h['fastvariabel'] == 'fast' else 'variable'
            print(f"  [{h['prioritet']}] {h['haeftelsestype']}: {h['hovedstol']}"
                  f" @ {rate_str}% ({fixed_var})")

            if h["alias"] in variable_aliases:
                isin = isin_map.get(h["prioritet"])
                _print_loan_type_estimate(float(rate_str), isin, alias=h.get("alias"))

            if h.get("kreditorer"):
                print(f"      Creditor(s): {', '.join(h['kreditorer'])}")

    if result.get("servitutter"):
        print("\nEasements:")
        for s in result["servitutter"]:
            print(f"  [{s['prioritet']}] {s['tekst']}")


if __name__ == "__main__":
    main()
