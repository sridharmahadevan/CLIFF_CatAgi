"""CLIFF-native culinary tour orchestration with conscious broadcasts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from .consciousness import ConsciousBroadcastBoard
from .textbook_backstop import recommend_textbook_backstop, render_textbook_backstop_html

_MONTH_PATTERN = (
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
)
_FOOD_FOCUS_STOPWORDS = (
    "a",
    "an",
    "and",
    "arrange",
    "below",
    "build",
    "create",
    "culinary",
    "design",
    "each",
    "every",
    "food",
    "for",
    "from",
    "in",
    "is",
    "less",
    "meal",
    "meals",
    "of",
    "orchestrate",
    "period",
    "plan",
    "the",
    "than",
    "to",
    "tour",
    "travel",
    "trip",
    "under",
    "where",
)
_DEFAULT_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_DEFAULT_OVERPASS_FALLBACK_URLS = (
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
)
_DEFAULT_CULINARY_USER_AGENT = "CLIFF/0.0.1 (culinary tour live lookup)"


@dataclass(frozen=True)
class CulinaryTourQueryPlan:
    """Parsed culinary-tour intent."""

    query: str
    normalized_query: str
    food_focus: str
    destination: str
    time_window: str
    budget_per_meal: int | None
    estimated_days: int


@dataclass(frozen=True)
class DiscoveredCulinaryStop:
    """A retrieved candidate stop for a culinary itinerary."""

    stop_id: str
    name: str
    destination: str
    district: str
    specialty: str
    estimated_cost: int | None
    tags: tuple[str, ...] = ()
    url: str = ""
    source_type: str = "manifest"
    score: float = 0.0


class CulinaryTourLookupError(RuntimeError):
    """Raised when the culinary route cannot recover real venue data."""


@dataclass(frozen=True)
class CulinaryTourRunResult:
    """Artifacts emitted by a culinary CLIFF run."""

    query_plan: CulinaryTourQueryPlan
    broadcasts_path: Path
    itinerary_path: Path
    dashboard_path: Path
    summary_path: Path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _slugify(text: str, *, maxlen: int = 80) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    cleaned = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return cleaned[:maxlen] or "culinary_tour"


def _extract_budget_per_meal(query: str) -> int | None:
    match = re.search(r"(?:under|below|less than)\s*\$?(\d+)\s*(?:per meal|a meal|meal)?", query, flags=re.I)
    if match:
        return int(match.group(1))
    return None


def _extract_time_window(query: str) -> str:
    match = re.search(
        rf"((?:from\s+)?{_MONTH_PATTERN}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*(?:[-–]|to)\s*\d{{1,2}}(?:st|nd|rd|th)?)?)",
        query,
        flags=re.I,
    )
    return " ".join(match.group(1).split()) if match else ""


def _estimate_day_count(time_window: str) -> int:
    match = re.search(r"(\d{1,2})\s*(?:[-–]|to)\s*(\d{1,2})", time_window)
    if not match:
        return 3
    start_day = int(match.group(1))
    end_day = int(match.group(2))
    if end_day >= start_day:
        return max(1, end_day - start_day + 1)
    return 3


def _extract_destination(query: str) -> str:
    match = re.search(
        rf"\b(?:in|to|around|of)\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*?)(?=\s+(?:from\s+)?{_MONTH_PATTERN}\b|\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*(?:[-–]|to)\s*\d{{1,2}}(?:st|nd|rd|th)?)?\b|\s+for\b|$)",
        query,
        flags=re.I,
    )
    if match:
        return match.group(1).strip()
    return "Destination TBD"


def _extract_food_focus(query: str, destination: str, time_window: str) -> str:
    quote_match = re.search(r"['\"]([^'\"]+)['\"]", query)
    if quote_match:
        return quote_match.group(1).strip()
    tour_match = re.search(
        r"\b(?:arrange|plan|build|design|create|orchestrate)\s+(?:me\s+)?(?:a|an|the\s+)?(.+?)\s+(?:culinary\s+|food\s+)?tour\b",
        query,
        flags=re.I,
    )
    if tour_match:
        candidate = " ".join(re.findall(r"[A-Za-z0-9]+", tour_match.group(1)))
        if candidate.strip():
            return candidate.strip()
    working = query
    if destination and destination != "Destination TBD":
        working = re.sub(re.escape(destination), " ", working, flags=re.I)
    if time_window:
        working = re.sub(re.escape(time_window), " ", working, flags=re.I)
    working = re.sub(r"\$\s*\d+", " ", working)
    working = re.sub(r"\b\d+\b", " ", working)
    working = re.sub(
        r"\b(?:plan|build|design|create|orchestrate|arrange|me|a|an|the|culinary|food|tour|travel|trip|itinerary|of|from|in|to|around|for|under|below|less|than|per|meal|meals|period|where|each|every|is)\b",
        " ",
        working,
        flags=re.I,
    )
    cleaned = " ".join(
        token
        for token in re.findall(r"[A-Za-z0-9]+", working)
        if token.lower() not in _FOOD_FOCUS_STOPWORDS
    )
    return cleaned.strip() or "local specialties"


def interpret_culinary_query(query: str) -> CulinaryTourQueryPlan:
    """Parse the culinary-tour request into a structured plan."""

    normalized_query = " ".join(query.lower().split())
    time_window = _extract_time_window(query)
    destination = _extract_destination(query)
    return CulinaryTourQueryPlan(
        query=" ".join(query.split()),
        normalized_query=normalized_query,
        food_focus=_extract_food_focus(query, destination, time_window),
        destination=destination,
        time_window=time_window,
        budget_per_meal=_extract_budget_per_meal(query),
        estimated_days=_estimate_day_count(time_window),
    )


def _default_culinary_manifest_path() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "culinary_tour" / "culinary_stop_manifest.jsonl"


def _destination_context(plan: CulinaryTourQueryPlan) -> dict[str, object]:
    return {
        "districts": [],
    }


def _price_label(cost: int | None) -> str:
    if cost is None:
        return "Price unavailable"
    return f"${cost}"


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", str(text).lower()))


def _coerce_tags(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(part).strip() for part in value if str(part).strip())
    return ()


def _load_manifest_records(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            values = payload.get("stops") or payload.get("venues") or []
            return [dict(item) for item in values if isinstance(item, dict)]
        raise ValueError(f"Unsupported culinary manifest payload in {path}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported culinary manifest format for {path}; expected .jsonl, .json, or .csv")


def _record_to_stop(record: dict[str, object], *, source_type: str) -> DiscoveredCulinaryStop:
    name = str(record.get("name") or record.get("title") or "Unnamed stop").strip()
    destination = str(record.get("destination") or record.get("city") or "Destination TBD").strip()
    district = str(record.get("district") or record.get("neighborhood") or "Unknown district").strip()
    specialty = str(record.get("specialty") or record.get("focus") or record.get("summary") or "local specialties").strip()
    estimated_cost_raw = record.get("estimated_cost") or record.get("price_estimate")
    estimated_cost = _coerce_estimated_cost(estimated_cost_raw)
    tags = _coerce_tags(record.get("tags") or record.get("cuisines") or record.get("keywords"))
    url = str(record.get("url") or "").strip()
    stop_id = str(record.get("stop_id") or record.get("id") or _slugify(f"{destination}_{district}_{name}", maxlen=48)).strip()
    return DiscoveredCulinaryStop(
        stop_id=stop_id,
        name=name,
        destination=destination,
        district=district,
        specialty=specialty,
        estimated_cost=estimated_cost,
        tags=tags,
        url=url,
        source_type=source_type,
    )


def _destination_matches(stop_destination: str, requested_destination: str) -> bool:
    stop_tokens = set(_tokenize(stop_destination))
    requested_tokens = set(_tokenize(requested_destination))
    if not stop_tokens or not requested_tokens:
        return False
    return stop_tokens <= requested_tokens or requested_tokens <= stop_tokens


def _coerce_estimated_cost(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    if set(text) <= {"$"}:
        return {
            1: 20,
            2: 35,
            3: 60,
            4: 95,
        }.get(len(text))
    return None


def _culinary_user_agent() -> str:
    configured = " ".join(os.environ.get("CLIFF_CULINARY_USER_AGENT", "").split()).strip()
    return configured or _DEFAULT_CULINARY_USER_AGENT


def _load_json_from_url(
    url: str,
    *,
    headers: dict[str, str],
    data: bytes | None = None,
    timeout: float = 20.0,
) -> object:
    request = urllib_request.Request(url, data=data, headers=headers)
    with urllib_request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _overpass_endpoints() -> tuple[str, ...]:
    configured = " ".join(os.environ.get("CLIFF_CULINARY_OVERPASS_URLS", "").split()).strip()
    if configured:
        candidates = tuple(item.strip() for item in configured.split(",") if item.strip())
        if candidates:
            return candidates
    explicit = " ".join(os.environ.get("CLIFF_CULINARY_OVERPASS_URL", "").split()).strip()
    if explicit:
        return (explicit,) + tuple(url for url in _DEFAULT_OVERPASS_FALLBACK_URLS if url != explicit)
    return (_DEFAULT_OVERPASS_URL,) + _DEFAULT_OVERPASS_FALLBACK_URLS


def _load_overpass_json(query: str) -> object:
    last_error: Exception | None = None
    endpoints = _overpass_endpoints()
    for attempt in range(3):
        for endpoint in endpoints:
            try:
                return _load_json_from_url(
                    endpoint,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                        "User-Agent": _culinary_user_agent(),
                    },
                    data=urllib_parse.urlencode({"data": query}).encode("utf-8"),
                    timeout=30.0,
                )
            except urllib_error.HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise CulinaryTourLookupError(
                        f"Live restaurant lookup failed via {endpoint}: HTTP {exc.code}"
                    ) from exc
            except urllib_error.URLError as exc:
                last_error = exc
            except TimeoutError as exc:
                last_error = exc
            except socket.timeout as exc:
                last_error = exc
        if attempt < 2:
            time.sleep(0.75 * (attempt + 1))
    detail = ""
    if isinstance(last_error, urllib_error.HTTPError):
        detail = f"HTTP {last_error.code}"
    elif isinstance(last_error, urllib_error.URLError):
        detail = str(last_error.reason)
    elif last_error is not None:
        detail = str(last_error)
    raise CulinaryTourLookupError(
        f"Live restaurant lookup failed after retrying {len(endpoints)} Overpass endpoint(s)"
        + (f": {detail}" if detail else ".")
    )


def _geocode_destination(destination: str) -> dict[str, object]:
    params = urllib_parse.urlencode(
        {
            "q": destination,
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 1,
        }
    )
    payload = _load_json_from_url(
        f"{os.environ.get('CLIFF_CULINARY_NOMINATIM_URL', _DEFAULT_NOMINATIM_URL)}?{params}",
        headers={
            "Accept": "application/json",
            "User-Agent": _culinary_user_agent(),
        },
    )
    if not isinstance(payload, list) or not payload:
        raise CulinaryTourLookupError(f"Could not geocode destination '{destination}'.")
    result = payload[0]
    if not isinstance(result, dict):
        raise CulinaryTourLookupError(f"Could not geocode destination '{destination}'.")
    return result


def _extract_focus_relevance(stop: DiscoveredCulinaryStop, food_focus: str) -> int:
    focus_tokens = set(_tokenize(food_focus))
    if not focus_tokens:
        return 0
    stop_tokens = set(_tokenize(" ".join((stop.name, stop.specialty, " ".join(stop.tags), stop.district))))
    return len(focus_tokens & stop_tokens)


def _district_from_tags(tags: dict[str, object], fallback_destination: str) -> str:
    for key in ("addr:suburb", "addr:neighbourhood", "addr:quarter", "addr:district", "addr:city_district", "addr:city"):
        value = str(tags.get(key) or "").strip()
        if value:
            return value
    return fallback_destination


def _url_from_tags(tags: dict[str, object]) -> str:
    for key in ("website", "contact:website", "url"):
        value = str(tags.get(key) or "").strip()
        if value:
            return value
    return ""


def _osm_element_to_stop(element: dict[str, object], *, destination: str) -> DiscoveredCulinaryStop | None:
    tags_obj = element.get("tags")
    if not isinstance(tags_obj, dict):
        return None
    tags = {str(key): value for key, value in tags_obj.items()}
    name = str(tags.get("name") or "").strip()
    if not name:
        return None
    specialty = str(tags.get("cuisine") or tags.get("description") or "restaurant").strip()
    estimated_cost = _coerce_estimated_cost(
        tags.get("price_range") or tags.get("price") or tags.get("charge")
    )
    osm_type = str(element.get("type") or "element")
    osm_id = str(element.get("id") or _slugify(name))
    venue_url = _url_from_tags(tags) or f"https://www.openstreetmap.org/{osm_type}/{osm_id}"
    return DiscoveredCulinaryStop(
        stop_id=f"osm_{osm_type}_{osm_id}",
        name=name,
        destination=destination,
        district=_district_from_tags(tags, destination),
        specialty=specialty,
        estimated_cost=estimated_cost,
        tags=_coerce_tags(tags.get("cuisine") or tags.get("diet:vegetarian") or tags.get("amenity")),
        url=venue_url,
        source_type="osm_live",
    )


def _live_stop_candidates(plan: CulinaryTourQueryPlan) -> list[DiscoveredCulinaryStop]:
    geocoded = _geocode_destination(plan.destination)
    bbox = geocoded.get("boundingbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise CulinaryTourLookupError(f"Destination '{plan.destination}' did not return a usable search area.")
    try:
        south, north, west, east = [float(item) for item in bbox]
    except (TypeError, ValueError) as exc:
        raise CulinaryTourLookupError(f"Destination '{plan.destination}' did not return a usable search area.") from exc

    overpass_query = (
        "[out:json][timeout:25];"
        "("
        f'nwr["amenity"~"restaurant|fast_food|food_court"]({south},{west},{north},{east});'
        ");"
        "out center tags;"
    )
    payload = _load_overpass_json(overpass_query)
    if not isinstance(payload, dict):
        raise CulinaryTourLookupError(f"Live restaurant lookup returned an invalid payload for '{plan.destination}'.")
    elements = payload.get("elements")
    if not isinstance(elements, list):
        raise CulinaryTourLookupError(f"Live restaurant lookup returned no elements for '{plan.destination}'.")
    stops = [
        stop
        for stop in (
            _osm_element_to_stop(element, destination=plan.destination)
            for element in elements
            if isinstance(element, dict)
        )
        if stop is not None
    ]
    if not stops:
        raise CulinaryTourLookupError(f"No restaurants were found near '{plan.destination}'.")
    relevant = [stop for stop in stops if _extract_focus_relevance(stop, plan.food_focus) > 0]
    if len(relevant) >= 6:
        return relevant
    return stops


def _score_stop(
    stop: DiscoveredCulinaryStop,
    *,
    plan: CulinaryTourQueryPlan,
    context: dict[str, object],
    budget_per_meal: int,
) -> float:
    focus_tokens = set(_tokenize(plan.food_focus))
    stop_tokens = set(_tokenize(" ".join((stop.name, stop.specialty, " ".join(stop.tags), stop.district))))
    destination_tokens = set(_tokenize(plan.destination))
    score = 0.0
    if set(_tokenize(stop.destination)) & destination_tokens:
        score += 4.0
    if stop.district in set(str(item) for item in context["districts"]):
        score += 1.0
    focus_overlap = focus_tokens & stop_tokens
    score += 1.6 * len(focus_overlap)
    if stop.estimated_cost is None:
        score -= 0.5
    elif stop.estimated_cost <= budget_per_meal:
        score += 1.8
    elif stop.estimated_cost <= budget_per_meal + 8:
        score += 0.9
    else:
        score -= min(1.5, (stop.estimated_cost - budget_per_meal) / 12.0)
    if stop.source_type in {"manifest", "osm_live"}:
        score += 0.4
    if stop.url:
        score += 0.2
    return round(score, 3)


def _select_top_stops(
    stops: list[DiscoveredCulinaryStop],
    *,
    plan: CulinaryTourQueryPlan,
    context: dict[str, object],
    budget_per_meal: int,
    limit: int = 6,
) -> list[DiscoveredCulinaryStop]:
    scored = [
        DiscoveredCulinaryStop(
            stop_id=stop.stop_id,
            name=stop.name,
            destination=stop.destination,
            district=stop.district,
            specialty=stop.specialty,
            estimated_cost=stop.estimated_cost,
            tags=stop.tags,
            url=stop.url,
            source_type=stop.source_type,
            score=_score_stop(stop, plan=plan, context=context, budget_per_meal=budget_per_meal),
        )
        for stop in stops
    ]
    ranked = sorted(scored, key=lambda item: (-item.score, item.estimated_cost, item.name))
    return ranked[:limit]


def _shortlist_retrieved_stops(
    stops: list[dict[str, object]],
    *,
    budget_per_meal: int | None,
    limit: int = 4,
) -> list[dict[str, object]]:
    shortlisted: list[dict[str, object]] = []
    seen_districts: set[str] = set()
    for stop in stops:
        district = str(stop.get("district") or "")
        estimated_cost = _coerce_estimated_cost(stop.get("estimated_cost"))
        if district in seen_districts and len(shortlisted) >= max(1, limit - 1):
            continue
        if budget_per_meal is not None:
            if estimated_cost is None:
                continue
            if estimated_cost > budget_per_meal:
                continue
        shortlisted.append(dict(stop))
        seen_districts.add(district)
        if len(shortlisted) >= limit:
            break
    return shortlisted


def _budget_diagnostics(stops: list[dict[str, object]], *, budget_per_meal: int | None) -> dict[str, int]:
    total = len(stops)
    with_known_price = 0
    under_budget = 0
    over_budget = 0
    missing_price = 0
    for stop in stops:
        estimated_cost = _coerce_estimated_cost(stop.get("estimated_cost"))
        if estimated_cost is None:
            missing_price += 1
            continue
        with_known_price += 1
        if budget_per_meal is None or estimated_cost <= budget_per_meal:
            under_budget += 1
        else:
            over_budget += 1
    return {
        "total": total,
        "with_known_price": with_known_price,
        "under_budget": under_budget,
        "over_budget": over_budget,
        "missing_price": missing_price,
    }


def _dashboard_html(
    *,
    plan: CulinaryTourQueryPlan,
    broadcasts: list[dict[str, object]],
    itinerary: list[dict[str, object]],
) -> str:
    def esc(value: object) -> str:
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    itinerary_cards = "".join(
        (
            '<article class="itinerary-card">'
            f'<div class="day-chip">{esc(item["day_label"])}</div>'
            f'<h3>{esc(item["venue_name"])}</h3>'
            f'<p class="district">{esc(item["district"])}</p>'
            f'<p>{esc(item["focus"])}</p>'
            f'<p class="cost">{esc(item["cost_label"])}</p>'
            + (
                f'<p class="trace"><a class="venue-link" href="{esc(item["venue_url"])}" target="_blank" rel="noopener noreferrer">More info</a></p>'
                if item.get("venue_url")
                else ""
            )
            + (
                f'<p class="trace">Guided by {esc(", ".join(item.get("supporting_broadcast_ids") or []))}</p>'
                if item.get("supporting_broadcast_ids")
                else ""
            )
            + "</article>"
        )
        for item in itinerary
    )
    broadcast_cards = "".join(
        (
            '<article class="broadcast-card">'
            f'<div class="broadcast-id">{esc(item["broadcast_id"])}</div>'
            f'<h3>{esc(item["title"])}</h3>'
            f'<p class="source">{esc(item["source_agent"])}</p>'
            f'<p>{esc(item["summary"])}</p>'
            + (
                f'<p class="trace">Read from {esc(", ".join(item.get("read_broadcast_ids") or []))}</p>'
                if item.get("read_broadcast_ids")
                else '<p class="trace">Read from none</p>'
            )
            + (
                f'<p class="trace">Tags: {esc(", ".join(item.get("tags") or []))}</p>'
                if item.get("tags")
                else ""
            )
            + (
                f'<p class="trace">Backend: {esc(item["payload"].get("retrieval_backend"))}</p>'
                if isinstance(item.get("payload"), dict) and item["payload"].get("retrieval_backend")
                else ""
            )
            + "</article>"
        )
        for item in broadcasts
    )
    budget_text = f"Under ${plan.budget_per_meal} per meal" if plan.budget_per_meal is not None else "Flexible budget"
    time_text = plan.time_window or "Dates flexible"
    textbook_html = render_textbook_backstop_html(
        recommend_textbook_backstop(plan.query, route_name="culinary_tour"),
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>CLIFF Culinary Tour</title>
    <style>
      :root {{
        --ink: #16211f;
        --muted: #5a6661;
        --paper: #f6f0e3;
        --card: rgba(255, 252, 246, 0.96);
        --line: #d7c7a6;
        --accent: #9a3f12;
        --accent-soft: #f7e6d6;
        --green: #1f6a48;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: Georgia, "Palatino Linotype", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(154,63,18,0.16), transparent 24%),
          radial-gradient(circle at top right, rgba(31,106,72,0.12), transparent 22%),
          linear-gradient(180deg, #fbf6eb 0%, var(--paper) 100%);
      }}
      main {{
        width: min(1200px, calc(100vw - 32px));
        margin: 34px auto 44px;
        display: grid;
        gap: 20px;
      }}
      .panel {{
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 28px;
        padding: 26px;
        box-shadow: 0 24px 60px rgba(45, 33, 15, 0.12);
      }}
      .eyebrow {{
        margin: 0 0 10px;
        text-transform: uppercase;
        letter-spacing: 0.17em;
        font-size: 12px;
        color: var(--accent);
      }}
      h1, h2, h3 {{ margin: 0; }}
      .hero-grid {{
        display: grid;
        gap: 16px;
        grid-template-columns: 1.4fr 1fr;
      }}
      .hero-query {{
        margin-top: 16px;
        font-size: 1.02rem;
        line-height: 1.6;
        color: var(--muted);
      }}
      .chip-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
      }}
      .chip {{
        border-radius: 999px;
        padding: 8px 12px;
        background: #efe7d6;
        color: #5f4421;
        font-size: 0.92rem;
      }}
      .section-grid {{
        display: grid;
        gap: 18px;
        grid-template-columns: 1.1fr 0.9fr;
      }}
      .itinerary-grid, .broadcast-grid {{
        display: grid;
        gap: 12px;
      }}
      .itinerary-card, .broadcast-card {{
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 16px;
        background: #fffdf8;
      }}
      .day-chip {{
        display: inline-flex;
        border-radius: 999px;
        padding: 6px 10px;
        background: #e6f2ea;
        color: var(--green);
        font-size: 0.82rem;
        margin-bottom: 10px;
      }}
      .district, .source {{
        margin: 8px 0 0;
        color: var(--muted);
        font-size: 0.94rem;
      }}
      .cost {{
        margin-top: 10px;
        color: var(--accent);
        font-weight: 700;
      }}
      .broadcast-id {{
        font-size: 0.76rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 10px;
      }}
      .venue-link {{
        color: var(--green);
        font-weight: 700;
        text-decoration: none;
      }}
      .venue-link:hover {{
        text-decoration: underline;
      }}
      .trace {{
        margin: 10px 0 0;
        color: var(--muted);
        line-height: 1.5;
        font-size: 0.9rem;
      }}
      .footnote {{
        margin-top: 14px;
        color: var(--muted);
        line-height: 1.6;
      }}
      .textbook-list {{
        padding-left: 20px;
        display: grid;
        gap: 10px;
      }}
      .mono {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 13px;
      }}
      @media (max-width: 860px) {{
        .hero-grid, .section-grid {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="panel hero-grid">
        <div>
          <p class="eyebrow">CLIFF Culinary Tour</p>
          <h1>{esc(plan.food_focus)} in {esc(plan.destination)}</h1>
          <p class="hero-query">{esc(plan.query)}</p>
          <div class="chip-row">
            <span class="chip">{esc(time_text)}</span>
            <span class="chip">{esc(budget_text)}</span>
            <span class="chip">{esc(plan.estimated_days)} planned days</span>
          </div>
        </div>
        <div class="panel" style="padding:18px; background: var(--accent-soft);">
          <p class="eyebrow">Why Consciousness Matters</p>
          <p class="footnote">Later unconscious agents read earlier conscious broadcasts before composing the itinerary. This turns the conscious layer into a shared message-passing workspace rather than a dead-end status screen.</p>
        </div>
      </section>
      <section class="panel">
        {textbook_html}
      </section>
      <section class="section-grid">
        <section class="panel">
          <p class="eyebrow">Proposed Itinerary</p>
          <div class="itinerary-grid">{itinerary_cards}</div>
        </section>
        <section class="panel">
          <p class="eyebrow">Conscious Broadcasts</p>
          <div class="broadcast-grid">{broadcast_cards}</div>
        </section>
      </section>
    </main>
  </body>
</html>"""


class CulinaryTourAgenticRunner:
    """Run a message-passing culinary tour through CLIFF's conscious workspace."""

    def __init__(self, query: str, outdir: str | Path, manifest_path: str | Path | None = None) -> None:
        self.query = " ".join(str(query).split()).strip()
        self.outdir = Path(outdir).resolve()
        self.manifest_path = Path(manifest_path).resolve() if manifest_path else None
        self.manifest_used_path: Path | None = None
        self.retrieval_backend = "seed_context"
        if not self.query:
            raise ValueError("A non-empty culinary tour query is required.")
        self.broadcast_board = ConsciousBroadcastBoard()

    def _resolve_manifest_path(self) -> Path | None:
        if self.manifest_path is not None:
            return self.manifest_path
        return None

    def _publish_intent_broadcast(self, plan: CulinaryTourQueryPlan) -> None:
        self.broadcast_board.publish(
            source_agent="intent_interpreter_agent",
            title="Normalized culinary intent",
            summary=(
                f"Detected a {plan.food_focus} tour in {plan.destination} "
                f"with time window '{plan.time_window or 'flexible'}'."
            ),
            payload=asdict(plan),
            tags=("intent", "constraints"),
        )

    def _publish_destination_context(self, plan: CulinaryTourQueryPlan) -> dict[str, object]:
        intent_messages = self.broadcast_board.messages_for_agent("destination_context_agent", tag="intent")
        context = _destination_context(plan)
        self.broadcast_board.publish(
            source_agent="destination_context_agent",
            title="Destination dining context",
            summary=(
                f"Prepared district context for {plan.destination} across "
                f"{', '.join(list(context['districts'])[:3])}."
            ),
            payload=context,
            tags=("destination_context",),
            read_broadcast_ids=tuple(item.broadcast_id for item in intent_messages),
        )
        return context

    def _publish_budget_guard(self, plan: CulinaryTourQueryPlan) -> dict[str, object]:
        constraint_messages = self.broadcast_board.messages_for_agent("budget_guard_agent", tag="constraints")
        payload = {
            "budget_per_meal": plan.budget_per_meal,
            "guidance": (
                "Keep every stop at or below the requested meal budget."
                if plan.budget_per_meal is not None
                else "No explicit meal budget was provided; prioritize strong venue matches with real listing data."
            ),
        }
        summary = (
            f"Set a working meal budget of ${plan.budget_per_meal} and filtered plans accordingly."
            if plan.budget_per_meal is not None
            else "No meal budget was specified, so the itinerary remains budget-flexible."
        )
        self.broadcast_board.publish(
            source_agent="budget_guard_agent",
            title="Budget guardrails",
            summary=summary,
            payload=payload,
            tags=("budget",),
            read_broadcast_ids=tuple(item.broadcast_id for item in constraint_messages),
        )
        return payload

    def _publish_stop_retrieval(
        self,
        plan: CulinaryTourQueryPlan,
        context: dict[str, object],
        budget: dict[str, object],
    ) -> list[dict[str, object]]:
        read_messages = (
            self.broadcast_board.messages_for_agent("stop_retrieval_agent", tag="intent")
            + self.broadcast_board.messages_for_agent("stop_retrieval_agent", tag="destination_context")
            + self.broadcast_board.messages_for_agent("stop_retrieval_agent", tag="budget")
        )
        catalog_path = self._resolve_manifest_path()
        if catalog_path is not None and catalog_path.exists():
            raw_records = _load_manifest_records(catalog_path)
            manifest_corpus = [
                _record_to_stop(record, source_type="manifest")
                for record in raw_records
            ]
            destination_matched = [
                stop
                for stop in manifest_corpus
                if _destination_matches(stop.destination, plan.destination)
            ]
            if destination_matched:
                self.manifest_used_path = catalog_path
                self.retrieval_backend = "manifest"
                corpus = destination_matched
            else:
                raise CulinaryTourLookupError(
                    f"Manifest {catalog_path} does not contain any venues for {plan.destination}."
                )
        else:
            self.manifest_used_path = None
            self.retrieval_backend = "osm_live"
            corpus = _live_stop_candidates(plan)

        raw_budget = budget.get("budget_per_meal")
        budget_per_meal = int(raw_budget) if raw_budget is not None else 40
        top_stops = _select_top_stops(corpus, plan=plan, context=context, budget_per_meal=budget_per_meal)
        if not top_stops:
            raise CulinaryTourLookupError(
                f"No real restaurant candidates were found for {plan.food_focus} in {plan.destination}."
            )
        payload = {
            "retrieval_backend": self.retrieval_backend,
            "catalog_path": str(self.manifest_used_path) if self.manifest_used_path else None,
            "venues": [asdict(item) for item in top_stops],
        }
        self.broadcast_board.publish(
            source_agent="stop_retrieval_agent",
            title="Retrieved culinary stop candidates",
            summary=(
                f"Retrieved {len(top_stops)} candidate stops for {plan.food_focus} "
                f"using the {self.retrieval_backend} backend."
            ),
            payload=payload,
            tags=("retrieved_stops", "venues"),
            read_broadcast_ids=tuple(dict.fromkeys(item.broadcast_id for item in read_messages)),
        )
        return payload["venues"]

    def _publish_venue_scout(self, plan: CulinaryTourQueryPlan, retrieved_stops: list[dict[str, object]]) -> list[dict[str, object]]:
        read_messages = (
            self.broadcast_board.messages_for_agent("venue_scout_agent", tag="retrieved_stops")
            + self.broadcast_board.messages_for_agent("venue_scout_agent", tag="budget")
        )
        budget_messages = self.broadcast_board.messages_for_agent("venue_scout_agent", tag="budget")
        budget_per_meal: int | None = None
        if budget_messages:
            raw_budget = budget_messages[-1].payload.get("budget_per_meal")
            if raw_budget is not None:
                budget_per_meal = int(raw_budget)
        venue_candidates = _shortlist_retrieved_stops(retrieved_stops, budget_per_meal=budget_per_meal)
        if len(venue_candidates) < 2:
            diagnostics = _budget_diagnostics(retrieved_stops, budget_per_meal=budget_per_meal)
            if budget_per_meal is None:
                raise CulinaryTourLookupError(
                    f"Could not assemble a culinary schedule from real venue data for {plan.destination}. "
                    f"Retrieved {diagnostics['total']} live candidate(s), "
                    f"{diagnostics['with_known_price']} with explicit price data, and "
                    f"{diagnostics['missing_price']} without usable price data."
                )
            raise CulinaryTourLookupError(
                f"Could not assemble a budget-compliant culinary schedule under ${budget_per_meal} per meal "
                f"for {plan.destination}. Retrieved {diagnostics['total']} live candidate(s), "
                f"{diagnostics['with_known_price']} with explicit price data, "
                f"{diagnostics['under_budget']} within budget, and "
                f"{diagnostics['missing_price']} without usable price data."
            )
        self.broadcast_board.publish(
            source_agent="venue_scout_agent",
            title="Shortlisted venue candidates",
            summary=(
                f"Selected {len(venue_candidates)} candidate stops oriented around "
                f"{plan.food_focus} after reading retrieval broadcasts."
            ),
            payload={"venues": venue_candidates},
            tags=("shortlist", "venues"),
            read_broadcast_ids=tuple(dict.fromkeys(item.broadcast_id for item in read_messages)),
        )
        return venue_candidates

    def _publish_itinerary(self, plan: CulinaryTourQueryPlan, venues: list[dict[str, object]]) -> list[dict[str, object]]:
        read_messages = (
            self.broadcast_board.messages_for_agent("itinerary_composer_agent", tag="shortlist")
            + self.broadcast_board.messages_for_agent("itinerary_composer_agent", tag="destination_context")
            + self.broadcast_board.messages_for_agent("itinerary_composer_agent", tag="budget")
        )
        read_broadcast_ids = tuple(dict.fromkeys(item.broadcast_id for item in read_messages))
        itinerary: list[dict[str, object]] = []
        meal_slots = max(2, min(len(venues), plan.estimated_days + 1))
        for index in range(meal_slots):
            venue = venues[index % len(venues)]
            itinerary.append(
                {
                    "day_label": f"Day {min(plan.estimated_days, index + 1)}",
                    "venue_name": venue["name"],
                    "district": venue["district"],
                    "focus": f"{plan.food_focus} emphasis: {venue['specialty']}",
                    "cost_label": _price_label(_coerce_estimated_cost(venue.get("estimated_cost"))),
                    "venue_url": str(venue.get("url") or "").strip(),
                    "supporting_broadcast_ids": list(read_broadcast_ids),
                }
            )
        self.broadcast_board.publish(
            source_agent="itinerary_composer_agent",
            title="Proposed culinary itinerary",
            summary=(
                f"Composed {len(itinerary)} meal stops after reading shortlist, destination, "
                f"and budget broadcasts."
            ),
            payload={"itinerary": itinerary},
            tags=("itinerary",),
            read_broadcast_ids=read_broadcast_ids,
        )
        return itinerary

    def run(self) -> CulinaryTourRunResult:
        self.outdir.mkdir(parents=True, exist_ok=True)
        plan = interpret_culinary_query(self.query)
        self._publish_intent_broadcast(plan)
        context = self._publish_destination_context(plan)
        budget = self._publish_budget_guard(plan)
        retrieved_stops = self._publish_stop_retrieval(plan, context, budget)
        venues = self._publish_venue_scout(plan, retrieved_stops)
        itinerary = self._publish_itinerary(plan, venues)

        broadcasts_payload = [asdict(item) for item in self.broadcast_board.broadcasts()]
        broadcasts_path = self.outdir / "conscious_broadcasts.json"
        itinerary_path = self.outdir / "culinary_itinerary.json"
        dashboard_path = self.outdir / "culinary_tour_dashboard.html"
        summary_path = self.outdir / "culinary_tour_summary.json"

        _write_json(broadcasts_path, {"broadcasts": broadcasts_payload})
        _write_json(
            itinerary_path,
            {
                "query_plan": asdict(plan),
                "retrieval_backend": self.retrieval_backend,
                "culinary_manifest_path": str(self.manifest_used_path) if self.manifest_used_path else None,
                "itinerary": itinerary,
            },
        )
        dashboard_path.write_text(
            _dashboard_html(plan=plan, broadcasts=broadcasts_payload, itinerary=itinerary),
            encoding="utf-8",
        )
        _write_json(
            summary_path,
            {
                "query_plan": asdict(plan),
                "broadcasts_path": str(broadcasts_path),
                "itinerary_path": str(itinerary_path),
                "dashboard_path": str(dashboard_path),
                "retrieval_backend": self.retrieval_backend,
                "culinary_manifest_path": str(self.manifest_used_path) if self.manifest_used_path else None,
                "n_broadcasts": len(broadcasts_payload),
                "n_itinerary_stops": len(itinerary),
            },
        )
        return CulinaryTourRunResult(
            query_plan=plan,
            broadcasts_path=broadcasts_path,
            itinerary_path=itinerary_path,
            dashboard_path=dashboard_path,
            summary_path=summary_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a CLIFF culinary tour from a natural-language request.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()
    result = CulinaryTourAgenticRunner(
        args.query,
        args.outdir,
        manifest_path=args.manifest or None,
    ).run()
    print(json.dumps({"dashboard_path": str(result.dashboard_path), "summary_path": str(result.summary_path)}, indent=2))


if __name__ == "__main__":
    main()
