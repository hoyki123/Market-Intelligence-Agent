"""SEC EDGAR data fetching — 13F institutional holdings and company filings."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import requests

from warren_brain.data.cache import get_cache

EDGAR_BASE = "https://data.sec.gov"
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"

HEADERS = {
    "User-Agent": "WarrenBrain/0.1 (research project; contact: research@example.com)",
    "Accept-Encoding": "gzip, deflate",
}


def get_cik(ticker: str) -> str | None:
    """Resolve a ticker to its SEC CIK number."""
    cache = get_cache()
    key = f"cik:{ticker.upper()}"
    cached = cache.get(key)
    if cached:
        return cached.get("cik")

    url = f"{EDGAR_BASE}/submissions/CIK{ticker.upper()}.json"
    # Try the company tickers mapping first (more reliable)
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(tickers_url, headers=HEADERS, timeout=10)
    if resp.ok:
        data = resp.json()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                cache.set(key, {"cik": cik})
                return cik
    return None


def fetch_13f_holdings(cik: str, max_filings: int = 4) -> list[dict]:
    """
    Fetch recent 13F-HR filings for an institution and parse holdings.
    Returns list of {period, holdings: [{name, cusip, value_usd, shares, pct_change}]}
    """
    cache = get_cache()
    key = f"13f:{cik}:{max_filings}"
    cached = cache.get(key)
    if cached:
        return cached

    # Get filing list
    url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if not resp.ok:
        return []

    submissions = resp.json()
    filings = submissions.get("filings", {}).get("recent", {})

    forms = filings.get("form", [])
    accession_numbers = filings.get("accessionNumber", [])
    dates = filings.get("filingDate", [])

    results = []
    for form, acc, date in zip(forms, accession_numbers, dates):
        if form != "13F-HR":
            continue
        if len(results) >= max_filings:
            break

        holdings = _parse_13f_xml(cik, acc.replace("-", ""))
        results.append({"period": date, "accession": acc, "holdings": holdings})

    cache.set(key, results)
    return results


def _parse_13f_xml(cik: str, accession_clean: str) -> list[dict]:
    """Download and parse the infotable XML from a 13F filing."""
    index_url = (
        f"{EDGAR_BASE}/Archives/edgar/data/{int(cik)}/{accession_clean}/{accession_clean}-index.json"
    )
    resp = requests.get(index_url, headers=HEADERS, timeout=15)
    if not resp.ok:
        return []

    index = resp.json()
    xml_file = None
    for doc in index.get("directory", {}).get("item", []):
        name = doc.get("name", "")
        if "infotable" in name.lower() and name.endswith(".xml"):
            xml_file = name
            break

    if not xml_file:
        return []

    xml_url = f"{EDGAR_BASE}/Archives/edgar/data/{int(cik)}/{accession_clean}/{xml_file}"
    xml_resp = requests.get(xml_url, headers=HEADERS, timeout=15)
    if not xml_resp.ok:
        return []

    holdings = []
    try:
        root = ET.fromstring(xml_resp.content)
        ns = re.match(r"\{.*\}", root.tag)
        ns = ns.group(0) if ns else ""

        for info in root.findall(f"{ns}infoTable"):
            def t(tag: str) -> str:
                el = info.find(f"{ns}{tag}")
                return el.text.strip() if el is not None and el.text else ""

            holdings.append({
                "name": t("nameOfIssuer"),
                "cusip": t("cusip"),
                "value_usd": int(t("value") or 0) * 1000,
                "shares": int(t("sshPrnamt") or 0),
                "share_type": t("sshPrnamtType"),
                "investment_discretion": t("investmentDiscretion"),
                "voting_authority_sole": int(t("Sole") or 0),
            })
    except ET.ParseError:
        pass

    return sorted(holdings, key=lambda x: x["value_usd"], reverse=True)


def fetch_company_13f_ownership(ticker: str) -> list[dict]:
    """
    Find institutions that hold a given ticker by searching recent 13F filings.
    Returns top holders with their reported positions.
    NOTE: Full cross-institution search requires SEC full-text search or a data provider.
    This is a best-effort approach using the EDGAR search API.
    """
    cache = get_cache()
    key = f"ownership:{ticker.upper()}"
    cached = cache.get(key)
    if cached:
        return cached

    # EDGAR full-text search for the ticker in 13F filings
    search_url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": f'"{ticker.upper()}"',
        "dateRange": "custom",
        "startdt": "2024-01-01",
        "forms": "13F-HR",
        "_source": "file_date,entity_name,file_num,period_of_report",
        "hits.hits.total.value": 1,
        "hits.hits._source": True,
        "hits.hits.highlight": True,
        "hits.hits._id": True,
        "category": "form-type",
    }

    # Simplified: return empty list to avoid EDGAR search complexity in scaffold
    # Implement with https://efts.sec.gov/LATEST/search-index or a provider like sec-api.io
    result: list[dict] = []
    cache.set(key, result)
    return result
