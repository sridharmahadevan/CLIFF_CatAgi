"""Query-style review retrieval and product-feedback analysis."""

from __future__ import annotations

import argparse
import csv
import json
import re
import webbrowser
from dataclasses import asdict, dataclass, field
from html import unescape
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .evidence_convergence import (
    EvidenceConvergenceAdapter,
    EvidenceConvergencePolicy,
    EvidenceConvergenceTracker,
)
from .dashboard_query_launcher import DashboardQueryLauncher, DashboardQueryLauncherConfig
from .product_feedback_agentic import (
    ProductFeedbackAgenticConfig,
    ProductFeedbackAgenticRunner,
    ProductFeedbackRunResult,
)
from .product_feedback_corpus_synthesis import (
    ProductFeedbackCorpusSynthesisResult,
    build_product_feedback_corpus_synthesis,
)
from .product_feedback_visualizations import bootstrap_product_feedback_dashboard

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "article",
    "best",
    "buy",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "long",
    "my",
    "of",
    "on",
    "or",
    "product",
    "review",
    "reviews",
    "sofa",
    "the",
    "these",
    "thoughts",
    "to",
    "vs",
    "what",
    "with",
    "year",
}

_QUERY_PREFIXES = (
    "how easy is it to assemble ",
    "how easy is to assemble ",
    "how hard is it to assemble ",
    "how hard is to assemble ",
    "how easy is it to set up ",
    "how easy is to set up ",
    "how easy is it to put together ",
    "how easy is to put together ",
    "how easy is it to drive ",
    "how easy is to drive ",
    "how comfortable is it to run with ",
    "how easy is it to run with ",
    "how comfortable is ",
    "how comfortable are ",
    "what do owners say about ",
    "what do reviews say about ",
    "what do people say about ",
)

_ASPECT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("assembly", ("assemble", "assembly", "set up", "setup", "put together", "installation")),
    ("comfort", ("comfortable", "comfort", "seat comfort", "long term comfort")),
    ("driving", ("drive", "driving", "handle", "handling")),
    ("running", ("run with", "running", "run")),
    ("durability", ("durability", "long term durability", "longevity", "holds up")),
    ("maintenance", ("maintenance", "cleaning", "clean", "washable covers", "wash")),
    ("returns", ("return risk", "returns", "return")),
)


def _slugify(name: str, maxlen: int = 80) -> str:
    collapsed = re.sub(r"\s+", " ", name.strip().lower())
    cleaned = re.sub(r"[^a-z0-9 _-]+", "", collapsed).strip().replace(" ", "_")
    return cleaned[:maxlen] if cleaned else "review"


def _read_records(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        raise ValueError(f"Expected a JSON list in {path}")
    if suffix == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(dict(json.loads(line)))
        return records
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported manifest format for {path}; expected .json, .jsonl, or .csv")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _tokenize(text: str) -> tuple[str, ...]:
    normalized = " ".join(text.lower().split())
    return tuple(
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if token not in _STOPWORDS and len(token) > 1
    )


def _strip_wrapping_article(text: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", text.strip(), flags=re.IGNORECASE)


def _normalize_spaces(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _detect_query_aspect(query: str) -> str:
    lowered = _normalize_spaces(query).lower()
    for label, patterns in _ASPECT_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return label
    return ""


def _extract_product_phrase(query: str) -> str:
    cleaned = _normalize_spaces(query).rstrip("?.! ")
    lowered = cleaned.lower()
    for prefix in _QUERY_PREFIXES:
        if lowered.startswith(prefix):
            remainder = cleaned[len(prefix):]
            if " for " in remainder.lower():
                remainder = re.split(r"\bfor\b", remainder, maxsplit=1, flags=re.IGNORECASE)[-1]
            return _strip_wrapping_article(remainder)
    if re.search(r"\bfor\b", lowered):
        candidate = re.split(r"\bfor\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[-1]
        return _strip_wrapping_article(candidate)
    if re.search(r"\bwith\b", lowered):
        candidate = re.split(r"\bwith\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[-1]
        return _strip_wrapping_article(candidate)
    return cleaned


def _build_retrieval_query(query: str, *, configured_product_name: str = "", configured_brand_name: str = "") -> tuple[str, str]:
    product_phrase = _normalize_spaces(" ".join(part for part in (configured_brand_name, configured_product_name) if part).strip())
    if not product_phrase:
        product_phrase = _extract_product_phrase(query)
    aspect = _detect_query_aspect(query)
    if aspect == "assembly":
        retrieval_query = f"{product_phrase} assembly reviews"
    elif aspect == "comfort":
        retrieval_query = f"{product_phrase} comfort reviews"
    elif aspect == "driving":
        retrieval_query = f"{product_phrase} driving reviews"
    elif aspect == "running":
        retrieval_query = f"{product_phrase} running reviews"
    elif aspect == "durability":
        retrieval_query = f"{product_phrase} durability reviews"
    elif aspect == "maintenance":
        retrieval_query = f"{product_phrase} maintenance reviews"
    elif aspect == "returns":
        retrieval_query = f"{product_phrase} return reviews"
    else:
        retrieval_query = product_phrase or _normalize_spaces(query)
    return _normalize_spaces(product_phrase), _normalize_spaces(retrieval_query)


def _match_score(plan: "ReviewQueryPlan", *texts: str) -> tuple[float, tuple[str, ...]]:
    haystack = " ".join(texts).lower()
    evidence = []
    score = 0.0
    if plan.normalized_query and plan.normalized_query in haystack:
        score += 5.0
        evidence.append("exact_query")
    matched = sorted({token for token in plan.keyword_tokens if token in haystack})
    score += float(len(matched))
    evidence.extend(matched)
    if "review" in haystack:
        score += 0.5
    if any(term in haystack for term in ("comfort", "comfortable", "returned", "return", "fit")):
        score += 0.5
    return score, tuple(evidence)


def _looks_like_html(path_or_url: str) -> bool:
    lowered = path_or_url.lower()
    return lowered.endswith(".html") or lowered.endswith(".htm") or "html" in lowered


def _payload_looks_like_html(payload: str) -> bool:
    snippet = payload[:1000].lower()
    return any(marker in snippet for marker in ("<!doctype html", "<html", "<body", "<article", "<main"))


def _extract_html_region(html_text: str) -> str:
    candidates = []
    for pattern in (
        r"(?is)<article\b[^>]*>(.*?)</article>",
        r"(?is)<main\b[^>]*>(.*?)</main>",
        r"(?is)<body\b[^>]*>(.*?)</body>",
    ):
        candidates.extend(re.findall(pattern, html_text))
    if not candidates:
        return html_text
    return max(candidates, key=len)


def _strip_html(html_text: str) -> str:
    text = _extract_html_region(html_text)
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<svg.*?>.*?</svg>", " ", text)
    text = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", text)
    text = re.sub(r"(?is)<(header|footer|nav|aside|form)\b[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    filtered = []
    for line in lines:
        lowered = line.lower()
        if not line:
            continue
        if lowered in {"skip to content", "search", "menu"}:
            continue
        if "refund policy" in lowered or "return policy" in lowered or "shipping policy" in lowered:
            continue
        if lowered.startswith("home »") or lowered.startswith("home >"):
            continue
        if len(line.split()) <= 2 and lowered in {"facebook", "twitter", "instagram", "youtube", "pinterest", "tiktok"}:
            continue
        filtered.append(line)
    cleaned = "\n".join(filtered)
    return cleaned.strip()


def _extract_article_text(payload: str, *, source_hint: str) -> str:
    if _looks_like_html(source_hint) or _payload_looks_like_html(payload):
        return _strip_html(payload)
    return " ".join(payload.split())


def _normalize_product_image_reference(
    raw_reference: str,
    *,
    source_reference: str = "",
    base_path: Path | None = None,
) -> str | None:
    candidate = unescape(str(raw_reference or "").strip())
    if not candidate:
        return None
    if candidate.startswith("data:"):
        return candidate
    if candidate.startswith("//"):
        return f"https:{candidate}"
    parsed = urlparse(candidate)
    if parsed.scheme in {"http", "https", "file"}:
        return candidate
    if source_reference:
        source_parsed = urlparse(source_reference)
        if source_parsed.scheme in {"http", "https"}:
            return urljoin(source_reference, candidate)
    try:
        path_candidate = Path(candidate)
        if not path_candidate.is_absolute():
            if base_path is not None:
                path_candidate = (base_path / path_candidate).resolve()
            elif source_reference and not urlparse(source_reference).scheme:
                path_candidate = (Path(source_reference).expanduser().resolve().parent / path_candidate).resolve()
            else:
                path_candidate = path_candidate.resolve()
        return path_candidate.as_uri()
    except (OSError, ValueError):
        return None


def _extract_meta_image_candidates(html_text: str) -> list[str]:
    candidates: list[str] = []
    for tag in re.findall(r"(?is)<meta\b[^>]*>", html_text):
        key_match = re.search(r'(?is)\b(?:property|name)\s*=\s*["\']([^"\']+)["\']', tag)
        content_match = re.search(r'(?is)\bcontent\s*=\s*["\']([^"\']+)["\']', tag)
        if not key_match or not content_match:
            continue
        key = key_match.group(1).strip().lower()
        if key in {"og:image", "og:image:secure_url", "twitter:image", "twitter:image:src"}:
            candidates.append(content_match.group(1).strip())
    return candidates


def _collect_json_ld_image_candidates(payload: object, *, into: list[str]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered in {"image", "imageurl", "thumbnail", "thumbnailurl"}:
                if isinstance(value, str):
                    into.append(value)
                elif isinstance(value, list):
                    for item in value:
                        _collect_json_ld_image_candidates(item, into=into)
                elif isinstance(value, dict):
                    for nested_key in ("url", "contentUrl", "@id"):
                        nested_value = value.get(nested_key)
                        if isinstance(nested_value, str):
                            into.append(nested_value)
                    _collect_json_ld_image_candidates(value, into=into)
            else:
                _collect_json_ld_image_candidates(value, into=into)
        return
    if isinstance(payload, list):
        for item in payload:
            _collect_json_ld_image_candidates(item, into=into)


def _extract_json_ld_image_candidates(html_text: str) -> list[str]:
    candidates: list[str] = []
    for raw_payload in re.findall(
        r'(?is)<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html_text,
    ):
        normalized_payload = raw_payload.strip()
        if not normalized_payload:
            continue
        try:
            parsed = json.loads(normalized_payload)
        except json.JSONDecodeError:
            continue
        _collect_json_ld_image_candidates(parsed, into=candidates)
    return candidates


def _extract_product_image_url(payload: str, *, source_reference: str) -> str | None:
    if not (_looks_like_html(source_reference) or _payload_looks_like_html(payload)):
        return None
    for raw_candidate in _extract_meta_image_candidates(payload) + _extract_json_ld_image_candidates(payload):
        normalized = _normalize_product_image_reference(
            raw_candidate,
            source_reference=source_reference,
        )
        if normalized:
            return normalized
    return None


def _resolve_query_for_main(args: argparse.Namespace) -> str:
    query = " ".join(str(args.query or "").split()).strip()
    if query:
        return query
    artifact_path = Path(args.outdir).resolve() / "product_feedback_run" / "corpus_synthesis" / "product_feedback_corpus_synthesis.html"
    with DashboardQueryLauncher(
        DashboardQueryLauncherConfig(
            title="Product Feedback Query Dashboard",
            subtitle=(
                "Describe the product question you want BAFFLE to investigate across retrieved reviews. "
                "Examples: 'How comfortable is the Lovesac sectional sofa?' or "
                "'What do owners say about long-term maintenance for the Dyson V15?'"
            ),
            query_label="Review retrieval query",
            query_placeholder=(
                "How comfortable is the Lovesac sectional sofa?\n"
                "or\n"
                "What do owners say about long-term durability of the Herman Miller Aeron?"
            ),
            submit_label="Launch Product Feedback Run",
            waiting_message=(
                "The query has been captured. BAFFLE will open the generated product-feedback dashboard when the run finishes."
            ),
            artifact_path=artifact_path,
        )
    ) as launcher:
        return launcher.wait_for_submission()


def _open_dashboard_artifact(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    try:
        webbrowser.open(path.resolve().as_uri(), new=1, autoraise=True)
    except Exception:
        pass


@dataclass(frozen=True)
class ReviewQueryPlan:
    """Interpreted query intent for product-review retrieval."""

    query: str
    normalized_query: str
    keyword_tokens: tuple[str, ...]
    target_documents: int
    product_name: str = ""
    retrieval_query: str = ""


@dataclass(frozen=True)
class DiscoveredReviewDocument:
    """Metadata for a candidate review page."""

    title: str
    score: float
    retrieval_backend: str = "unknown"
    source_path: str | None = None
    url: str | None = None
    image_url: str | None = None
    abstract: str = ""
    evidence: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MaterializedReviewDocument:
    """A review page materialized into the working corpus."""

    title: str
    materialized_path: str
    extracted_text_path: str
    source_reference: str
    score: float
    retrieval_backend: str
    product_image_url: str | None = None


@dataclass(frozen=True)
class ProductFeedbackQueryAgenticConfig:
    """Configuration for query-style product-review retrieval plus analysis."""

    query: str
    outdir: Path
    manifest_path: Path | None = None
    retrieval_backend: str = "auto"
    retrieval_user_agent: str = "FunctorFlow_v2/0.1 (product review retrieval; local use)"
    retrieval_timeout_seconds: float = 20.0
    target_documents: int = 5
    max_documents: int = 0
    consensus_enabled: bool = True
    consensus_batch_size: int = 1
    consensus_score_tolerance: float = 0.03
    consensus_required_stable_passes: int = 1
    product_name: str = ""
    brand_name: str = ""
    analysis_question: str = ""
    discovery_only: bool = False
    enable_corpus_synthesis: bool = True

    def resolved(self) -> "ProductFeedbackQueryAgenticConfig":
        return ProductFeedbackQueryAgenticConfig(
            query=self.query.strip(),
            outdir=self.outdir.resolve(),
            manifest_path=self.manifest_path.resolve() if self.manifest_path else None,
            retrieval_backend=self.retrieval_backend,
            retrieval_user_agent=self.retrieval_user_agent,
            retrieval_timeout_seconds=self.retrieval_timeout_seconds,
            target_documents=self.target_documents,
            max_documents=self.max_documents,
            consensus_enabled=self.consensus_enabled,
            consensus_batch_size=self.consensus_batch_size,
            consensus_score_tolerance=self.consensus_score_tolerance,
            consensus_required_stable_passes=self.consensus_required_stable_passes,
            product_name=self.product_name.strip(),
            brand_name=self.brand_name.strip(),
            analysis_question=self.analysis_question.strip(),
            discovery_only=self.discovery_only,
            enable_corpus_synthesis=self.enable_corpus_synthesis,
        )


@dataclass(frozen=True)
class ProductFeedbackQueryRunResult:
    """Result bundle for review retrieval plus product-feedback analysis."""

    query_plan: ReviewQueryPlan
    selected_documents: tuple[DiscoveredReviewDocument, ...]
    materialized_documents: tuple[MaterializedReviewDocument, ...]
    materialized_feedback_manifest_path: Path
    summary_path: Path
    analysis_iterations: int = 0
    consensus_reached: bool = False
    convergence_assessment: dict[str, object] | None = None
    product_feedback_result: ProductFeedbackRunResult | None = None
    corpus_synthesis_result: ProductFeedbackCorpusSynthesisResult | None = None


@dataclass(frozen=True)
class ReviewConsensusSnapshot:
    """Compact view of the evolving product-feedback verdict."""

    verdict: str
    overall_score: float
    return_warning_recommended: bool
    top_positive_aspects: tuple[str, ...]
    top_negative_aspects: tuple[str, ...]
    top_return_risk_aspects: tuple[str, ...]
    feedback_count: int


class ProductFeedbackConvergenceAdapter(EvidenceConvergenceAdapter[ReviewConsensusSnapshot]):
    """Convergence semantics for evolving product-feedback verdicts."""

    def __init__(self, *, score_tolerance: float) -> None:
        self.score_tolerance = max(0.0, float(score_tolerance))

    def similarity(
        self,
        previous: ReviewConsensusSnapshot,
        current: ReviewConsensusSnapshot,
        *,
        policy: EvidenceConvergencePolicy,
    ) -> float:
        del policy
        if previous.verdict != current.verdict:
            return 0.0
        if previous.return_warning_recommended != current.return_warning_recommended:
            return 0.0
        if abs(previous.overall_score - current.overall_score) > self.score_tolerance:
            return 0.0
        if not self._stable_aspect_overlap(previous.top_positive_aspects, current.top_positive_aspects):
            return 0.0
        if not self._stable_aspect_overlap(previous.top_negative_aspects, current.top_negative_aspects):
            return 0.0
        if not self._stable_aspect_overlap(previous.top_return_risk_aspects, current.top_return_risk_aspects):
            return 0.0
        return 1.0

    def describe(self, snapshot: ReviewConsensusSnapshot) -> str:
        return (
            f"verdict={snapshot.verdict}, score={snapshot.overall_score:.3f}, "
            f"feedback_count={snapshot.feedback_count}"
        )

    @staticmethod
    def _stable_aspect_overlap(previous: tuple[str, ...], current: tuple[str, ...]) -> bool:
        if not previous and not current:
            return True
        left = set(previous[:2])
        right = set(current[:2])
        if not left or not right:
            return False
        return bool(left & right)


class ReviewRetrievalBackend(Protocol):
    """Review-retrieval backend interface."""

    backend_name: str

    def search(self, plan: ReviewQueryPlan, *, limit: int) -> tuple[DiscoveredReviewDocument, ...]:
        """Return ranked review documents for a product query."""


class WebSearchReviewRetrievalBackend:
    """Search the public web for likely review pages and return ranked links."""

    backend_name = "web_search"

    def __init__(self, *, user_agent: str, timeout_seconds: float) -> None:
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds

    def search(self, plan: ReviewQueryPlan, *, limit: int) -> tuple[DiscoveredReviewDocument, ...]:
        html_text = self._fetch_search_html(self._search_query(plan), limit=max(limit, plan.target_documents))
        candidates = self._parse_search_results(plan, html_text)
        candidates.sort(key=lambda item: (-item.score, item.title.lower()))
        deduped: list[DiscoveredReviewDocument] = []
        seen_urls: set[str] = set()
        for item in candidates:
            key = (item.url or item.source_path or item.title).strip().lower()
            if key in seen_urls:
                continue
            seen_urls.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return tuple(deduped)

    def _search_query(self, plan: ReviewQueryPlan) -> str:
        normalized = " ".join((plan.retrieval_query or plan.query).split())
        lowered = normalized.lower()
        if "review" in lowered or "reviews" in lowered:
            return normalized
        return f"{normalized} reviews"

    def _fetch_search_html(self, query: str, *, limit: int) -> str:
        request = Request(
            f"https://html.duckduckgo.com/html/?{urlencode({'q': query, 'kl': 'us-en'})}",
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            payload = response.read().decode(charset, errors="replace")
        del limit
        return payload

    def _parse_search_results(
        self,
        plan: ReviewQueryPlan,
        payload: str,
    ) -> list[DiscoveredReviewDocument]:
        candidates: list[DiscoveredReviewDocument] = []
        blocks = re.findall(r'(?is)<div[^>]*class="[^"]*result[^"]*"[^>]*>(.*?)</div>\s*</div>?', payload)
        if not blocks:
            blocks = re.findall(r'(?is)<a[^>]*class="[^"]*result__a[^"]*"[^>]*>.*?</a>.*?(?=<a[^>]*class="[^"]*result__a|\Z)', payload)
        for block in blocks:
            link_match = re.search(r'(?is)<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block)
            if not link_match:
                continue
            raw_href, raw_title = link_match.groups()
            url = self._decode_result_url(raw_href)
            if not url:
                continue
            title = self._clean_html_fragment(raw_title)
            snippet_match = re.search(r'(?is)<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', block)
            if not snippet_match:
                snippet_match = re.search(r'(?is)<div[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</div>', block)
            abstract = self._clean_html_fragment(snippet_match.group(1)) if snippet_match else ""
            score, evidence = _match_score(plan, title, abstract, url)
            if any(token in f"{title} {abstract}".lower() for token in ("review", "comfortable", "comfort", "durability", "returned", "return")):
                score += 1.0
            parsed_url = urlparse(url)
            hostname = parsed_url.netloc.lower()
            if any(token in hostname for token in ("review", "gear", "guide", "guru", "tester", "castle")):
                score += 0.35
            if score <= 0.0:
                continue
            candidates.append(
                DiscoveredReviewDocument(
                    title=title or _slugify(url),
                    score=score,
                    retrieval_backend=self.backend_name,
                    url=url,
                    abstract=abstract,
                    evidence=evidence,
                    metadata={"host": hostname},
                )
            )
        return candidates

    @staticmethod
    def _decode_result_url(raw_href: str) -> str:
        href = unescape(raw_href).strip()
        if href.startswith("//"):
            href = "https:" + href
        parsed = urlparse(href)
        if "duckduckgo.com" in parsed.netloc:
            encoded = parse_qs(parsed.query).get("uddg", [""])[0]
            if encoded:
                return encoded
        return href

    @staticmethod
    def _clean_html_fragment(fragment: str) -> str:
        text = re.sub(r"(?s)<[^>]+>", " ", fragment)
        return " ".join(unescape(text).split())


class ManifestReviewRetrievalBackend:
    """Search a local manifest of review URLs or paths using token overlap."""

    backend_name = "manifest"

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path
        self._records = _read_records(manifest_path)

    def search(self, plan: ReviewQueryPlan, *, limit: int) -> tuple[DiscoveredReviewDocument, ...]:
        candidates: list[DiscoveredReviewDocument] = []
        for record in self._records:
            title = str(record.get("title") or record.get("name") or "").strip()
            abstract = str(record.get("abstract") or record.get("summary") or record.get("description") or "").strip()
            keywords = str(record.get("keywords") or record.get("tags") or "").strip()
            product = str(record.get("product") or "").strip()
            brand = str(record.get("brand") or "").strip()
            image_url = _normalize_product_image_reference(
                str(
                    record.get("image_url")
                    or record.get("image")
                    or record.get("thumbnail_url")
                    or record.get("thumbnail")
                    or ""
                ),
                base_path=self.manifest_path.parent,
            )
            score, evidence = _match_score(plan, title, abstract, keywords, product, brand)
            if score <= 0.0:
                continue
            source_path = self._resolve_source_path(record)
            url = str(record.get("url") or record.get("review_url") or "").strip() or None
            candidates.append(
                DiscoveredReviewDocument(
                    title=title or (Path(source_path).stem if source_path else _slugify(url or "review")),
                    score=score,
                    retrieval_backend=self.backend_name,
                    source_path=str(source_path) if source_path else None,
                    url=url,
                    image_url=image_url,
                    abstract=abstract,
                    evidence=evidence,
                    metadata={
                        key: str(value)
                        for key, value in record.items()
                        if key not in {"title", "name", "abstract", "summary", "description", "keywords", "tags", "url", "review_url", "source_path", "path"}
                        and value not in (None, "")
                    },
                )
            )
        candidates.sort(key=lambda item: (-item.score, item.title.lower()))
        return tuple(candidates[:limit])

    def _resolve_source_path(self, record: dict[str, object]) -> Path | None:
        for key in ("source_path", "path", "local_path", "file_path"):
            raw = record.get(key)
            if not raw:
                continue
            candidate = Path(str(raw))
            if candidate.is_absolute():
                return candidate
            return (self.manifest_path.parent / candidate).resolve()
        return None


class ProductFeedbackQueryAgenticRunner:
    """Run product-review retrieval and then feed results into product-feedback analysis."""

    def __init__(self, config: ProductFeedbackQueryAgenticConfig) -> None:
        self.config = config.resolved()
        self.summary_path = self.config.outdir / "review_query_summary.json"
        self.discovery_manifest_path = self.config.outdir / "discovered_review_manifest.jsonl"
        self.materialized_feedback_manifest_path = self.config.outdir / "materialized_feedback_manifest.jsonl"
        self.materialization_failures: list[dict[str, str]] = []
        self.analysis_outdir = self.config.outdir / "product_feedback_run"
        self.product_visual_asset_path = self.analysis_outdir / "product_visual_asset.json"

    def run(self) -> ProductFeedbackQueryRunResult:
        self.config.outdir.mkdir(parents=True, exist_ok=True)
        analysis_question = self.config.analysis_question or (
            f"What do retrieved reviews suggest about whether {self.config.product_name or 'this product'} "
            "is successful, comfortable over the long run, or at risk for returns?"
        )
        self._bootstrap_feedback_dashboard(
            status="starting",
            note="Preparing retrieval plan and dashboard.",
            feedback_count=0,
            analysis_question=analysis_question,
        )
        plan = self._run_query_interpretation_agent()
        selected = self._run_review_discovery_agent(plan)
        materialized: tuple[MaterializedReviewDocument, ...] = ()
        feedback_result = None
        analysis_iterations = 0
        consensus_reached = False
        convergence_assessment = None
        if self.config.discovery_only:
            selected = tuple(selected[: plan.target_documents])
            self._refresh_product_visual_asset(discovered=selected)
            materialized = self._run_review_materialization_agent(selected)
            self._refresh_product_visual_asset(discovered=selected, materialized=materialized)
            feedback_manifest = self._run_feedback_extraction_agent(materialized)
            self._bootstrap_feedback_dashboard(
                status="discovery_complete",
                note=f"Collected {len(materialized)} review sources for inspection.",
                feedback_count=len(materialized),
                analysis_question=analysis_question,
            )
        else:
            (
                selected,
                materialized,
                feedback_manifest,
                feedback_result,
                analysis_iterations,
                consensus_reached,
                convergence_assessment,
            ) = (
                self._run_iterative_feedback_loop(
                    discovered=selected,
                    analysis_question=analysis_question,
                )
            )

        result = ProductFeedbackQueryRunResult(
            query_plan=plan,
            selected_documents=selected,
            materialized_documents=materialized,
            materialized_feedback_manifest_path=feedback_manifest,
            summary_path=self.summary_path,
            analysis_iterations=analysis_iterations,
            consensus_reached=consensus_reached,
            convergence_assessment=convergence_assessment,
            product_feedback_result=feedback_result,
            corpus_synthesis_result=(
                build_product_feedback_corpus_synthesis(
                    query=self.config.query,
                    outdir=self.analysis_outdir,
                    feedback_result=feedback_result,
                )
                if feedback_result is not None and self.config.enable_corpus_synthesis
                else None
            ),
        )
        self.summary_path.write_text(
            json.dumps(
                {
                    "query_plan": asdict(plan),
                    "selected_documents": [asdict(item) for item in selected],
                    "discovery_manifest_path": str(self.discovery_manifest_path),
                    "materialized_documents": [asdict(item) for item in materialized],
                    "materialization_failures": list(self.materialization_failures),
                    "materialized_feedback_manifest_path": str(feedback_manifest),
                    "product_visual_asset_path": (
                        str(self.product_visual_asset_path) if self.product_visual_asset_path.exists() else None
                    ),
                    "analysis_iterations": analysis_iterations,
                    "consensus_reached": consensus_reached,
                    "convergence_assessment": convergence_assessment,
                    "product_feedback_report_path": str(feedback_result.report_path) if feedback_result else None,
                    "product_feedback_dashboard_path": str(feedback_result.dashboard_path) if feedback_result else None,
                    "corpus_synthesis_summary_path": (
                        str(result.corpus_synthesis_result.summary_path) if result.corpus_synthesis_result else None
                    ),
                    "corpus_synthesis_dashboard_path": (
                        str(result.corpus_synthesis_result.dashboard_path) if result.corpus_synthesis_result else None
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return result

    def _run_query_interpretation_agent(self) -> ReviewQueryPlan:
        query = " ".join(self.config.query.split())
        product_name, retrieval_query = _build_retrieval_query(
            query,
            configured_product_name=self.config.product_name,
            configured_brand_name=self.config.brand_name,
        )
        return ReviewQueryPlan(
            query=query,
            normalized_query=(retrieval_query or query).lower(),
            keyword_tokens=_tokenize(" ".join(part for part in (query, product_name, retrieval_query) if part)),
            target_documents=max(int(self.config.target_documents), 1),
            product_name=product_name,
            retrieval_query=retrieval_query,
        )

    def _run_review_discovery_agent(self, plan: ReviewQueryPlan) -> tuple[DiscoveredReviewDocument, ...]:
        backend = self._resolve_backend()
        discovered = backend.search(plan, limit=self._discovery_limit(plan))
        _write_jsonl(
            self.discovery_manifest_path,
            [
                {
                    "title": item.title,
                    "url": item.url,
                    "source_path": item.source_path,
                    "summary": item.abstract,
                    "retrieval_backend": item.retrieval_backend,
                    "score": item.score,
                    "image_url": item.image_url,
                    "evidence": list(item.evidence),
                    "metadata": item.metadata,
                }
                for item in discovered
            ],
        )
        return discovered

    def _discovery_limit(self, plan: ReviewQueryPlan) -> int:
        if self.config.discovery_only:
            return plan.target_documents
        requested = self.config.max_documents if self.config.max_documents > 0 else max(plan.target_documents * 4, 20)
        return max(plan.target_documents, requested)

    def _max_documents_to_consider(self, discovered_count: int) -> int:
        if self.config.max_documents > 0:
            return min(self.config.max_documents, discovered_count)
        return discovered_count

    def _run_iterative_feedback_loop(
        self,
        *,
        discovered: tuple[DiscoveredReviewDocument, ...],
        analysis_question: str,
    ) -> tuple[
        tuple[DiscoveredReviewDocument, ...],
        tuple[MaterializedReviewDocument, ...],
        Path,
        ProductFeedbackRunResult | None,
        int,
        bool,
        dict[str, object] | None,
    ]:
        raw_dir = self.config.outdir / "materialized_reviews" / "raw"
        text_dir = self.config.outdir / "materialized_reviews" / "text"
        raw_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)
        selected: list[DiscoveredReviewDocument] = []
        materialized: list[MaterializedReviewDocument] = []
        feedback_result: ProductFeedbackRunResult | None = None
        analysis_iterations = 0
        consensus_reached = False
        convergence_assessment: dict[str, object] | None = None
        max_documents = self._max_documents_to_consider(len(discovered))
        batch_size = max(1, int(self.config.consensus_batch_size))
        convergence_tracker = self._build_convergence_tracker(max_documents=max_documents)
        self._refresh_product_visual_asset(discovered=discovered[:max_documents])
        self._bootstrap_feedback_dashboard(
            status="discovering_reviews",
            note="Searching for candidate review evidence.",
            feedback_count=0,
            analysis_question=analysis_question,
        )
        for candidate_index, document in enumerate(discovered[:max_documents], start=1):
            materialized_doc = self._materialize_review_document(
                document,
                index=len(selected) + 1,
                raw_dir=raw_dir,
                text_dir=text_dir,
            )
            if materialized_doc is None:
                continue
            selected.append(document)
            materialized.append(materialized_doc)
            self._refresh_product_visual_asset(discovered=tuple(selected), materialized=tuple(materialized))
            feedback_manifest = self._run_feedback_extraction_agent(tuple(materialized))
            self._bootstrap_feedback_dashboard(
                status="collecting_evidence",
                note=f"Collected {len(materialized)} review sources. Updating the product verdict as evidence accumulates.",
                feedback_count=len(materialized),
                analysis_question=analysis_question,
            )
            if len(materialized) < self.config.target_documents:
                continue
            is_batch_boundary = (len(materialized) - self.config.target_documents) % batch_size == 0
            is_last_candidate = candidate_index == max_documents
            if not is_batch_boundary and not is_last_candidate:
                continue
            feedback_result = self._run_product_feedback_analysis(feedback_manifest, analysis_question=analysis_question)
            analysis_iterations += 1
            current_snapshot = self._consensus_snapshot(feedback_result)
            assessment = convergence_tracker.assess(
                current_snapshot,
                evidence_count=len(materialized),
            )
            convergence_assessment = assessment.as_dict()
            if self.config.consensus_enabled and assessment.stop:
                consensus_reached = assessment.stop_trigger == "stability"
                break
        if not materialized:
            preview = "; ".join(
                f"{item['title']}: {item['error']}" for item in self.materialization_failures[:3]
            ) or "no documents could be materialized"
            raise RuntimeError(f"Failed to materialize any review pages. First issues: {preview}")
        feedback_manifest = self._run_feedback_extraction_agent(tuple(materialized))
        if feedback_result is None:
            feedback_result = self._run_product_feedback_analysis(feedback_manifest, analysis_question=analysis_question)
            analysis_iterations += 1
        return (
            tuple(selected),
            tuple(materialized),
            feedback_manifest,
            feedback_result,
            analysis_iterations,
            consensus_reached,
            convergence_assessment,
        )

    def _run_review_materialization_agent(
        self, documents: tuple[DiscoveredReviewDocument, ...]
    ) -> tuple[MaterializedReviewDocument, ...]:
        raw_dir = self.config.outdir / "materialized_reviews" / "raw"
        text_dir = self.config.outdir / "materialized_reviews" / "text"
        raw_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)

        materialized: list[MaterializedReviewDocument] = []
        for index, document in enumerate(documents, start=1):
            materialized_doc = self._materialize_review_document(document, index=index, raw_dir=raw_dir, text_dir=text_dir)
            if materialized_doc is not None:
                materialized.append(materialized_doc)
        if not materialized:
            preview = "; ".join(
                f"{item['title']}: {item['error']}" for item in self.materialization_failures[:3]
            ) or "no documents could be materialized"
            raise RuntimeError(f"Failed to materialize any review pages. First issues: {preview}")
        return tuple(materialized)

    def _materialize_review_document(
        self,
        document: DiscoveredReviewDocument,
        *,
        index: int,
        raw_dir: Path,
        text_dir: Path,
    ) -> MaterializedReviewDocument | None:
        stem = f"{index:04d}_{_slugify(document.title)}"
        try:
            payload, source_reference, suffix = self._materialize_payload(document)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            self.materialization_failures.append(
                {
                    "title": document.title,
                    "source_reference": str(document.url or document.source_path or ""),
                    "error": str(exc),
                }
            )
            return None
        raw_path = raw_dir / f"{stem}{suffix}"
        raw_path.write_text(payload, encoding="utf-8")
        extracted_text = _extract_article_text(payload, source_hint=source_reference)
        if not extracted_text.strip():
            self.materialization_failures.append(
                {
                    "title": document.title,
                    "source_reference": source_reference,
                    "error": "extracted text was empty",
                }
            )
            return None
        text_path = text_dir / f"{stem}.txt"
        text_path.write_text(extracted_text, encoding="utf-8")
        return MaterializedReviewDocument(
            title=document.title,
            materialized_path=str(raw_path),
            extracted_text_path=str(text_path),
            source_reference=source_reference,
            score=document.score,
            retrieval_backend=document.retrieval_backend,
            product_image_url=document.image_url or _extract_product_image_url(payload, source_reference=source_reference),
        )

    def _run_feedback_extraction_agent(
        self, documents: tuple[MaterializedReviewDocument, ...]
    ) -> Path:
        rows = []
        for index, document in enumerate(documents, start=1):
            text = Path(document.extracted_text_path).read_text(encoding="utf-8").strip()
            rows.append(
                {
                    "id": f"retrieved_review_{index:04d}",
                    "title": document.title,
                    "text": text,
                    "source": document.retrieval_backend,
                    "source_reference": document.source_reference,
                    "retrieval_score": document.score,
                    "image_url": document.product_image_url,
                    "image_alt": self._product_visual_alt_text(),
                    "image_source_reference": document.source_reference,
                }
            )
        _write_jsonl(self.materialized_feedback_manifest_path, rows)
        return self.materialized_feedback_manifest_path

    def _product_visual_alt_text(self) -> str:
        label = " ".join(
            part
            for part in (
                self.config.brand_name.strip(),
                self.config.product_name.strip(),
            )
            if part
        ).strip()
        if not label:
            label = "product"
        return f"{label} visual"

    def _dashboard_product_label(self) -> str:
        label = self.config.product_name.strip()
        if label:
            return label
        query = " ".join(self.config.query.split()).strip()
        return query or "product"

    def _refresh_product_visual_asset(
        self,
        *,
        discovered: tuple[DiscoveredReviewDocument, ...] = (),
        materialized: tuple[MaterializedReviewDocument, ...] = (),
    ) -> None:
        image_url = ""
        source_reference = ""
        source_type = ""
        for document in materialized:
            if document.product_image_url:
                image_url = document.product_image_url
                source_reference = document.source_reference
                source_type = "materialized_review"
                break
        if not image_url:
            for document in discovered:
                if document.image_url:
                    image_url = document.image_url
                    source_reference = str(document.url or document.source_path or "")
                    source_type = "discovered_review"
                    break
        if not image_url:
            return
        payload = {
            "image_url": image_url,
            "image_alt": self._product_visual_alt_text(),
            "source_reference": source_reference,
            "source_type": source_type,
        }
        if self.product_visual_asset_path.exists():
            try:
                existing = json.loads(self.product_visual_asset_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
            if existing == payload:
                return
        _write_json(self.product_visual_asset_path, payload)

    def _run_product_feedback_analysis(
        self,
        feedback_manifest: Path,
        *,
        analysis_question: str,
    ) -> ProductFeedbackRunResult:
        return ProductFeedbackAgenticRunner(
            ProductFeedbackAgenticConfig(
                manifest_path=feedback_manifest,
                outdir=self.analysis_outdir,
                product_name=self._dashboard_product_label(),
                brand_name=self.config.brand_name,
                analysis_question=analysis_question,
            )
        ).run()

    def _bootstrap_feedback_dashboard(
        self,
        *,
        status: str,
        note: str,
        feedback_count: int,
        analysis_question: str,
    ) -> None:
        bootstrap_product_feedback_dashboard(
            self.analysis_outdir,
            product_name=self._dashboard_product_label(),
            brand_name=self.config.brand_name,
            analysis_question=analysis_question,
            run_status=status,
            run_status_note=note,
            feedback_count=feedback_count,
        )

    def _build_convergence_tracker(
        self,
        *,
        max_documents: int,
    ) -> EvidenceConvergenceTracker[ReviewConsensusSnapshot]:
        return EvidenceConvergenceTracker(
            policy=EvidenceConvergencePolicy(
                min_evidence=max(1, self.config.target_documents),
                stability_threshold=1.0,
                required_stable_passes=max(1, self.config.consensus_required_stable_passes),
                max_evidence=max_documents,
            ),
            adapter=ProductFeedbackConvergenceAdapter(
                score_tolerance=self.config.consensus_score_tolerance,
            ),
        )

    def _consensus_snapshot(self, feedback_result: ProductFeedbackRunResult) -> ReviewConsensusSnapshot:
        scorecard = json.loads(feedback_result.success_scorecard_path.read_text(encoding="utf-8"))
        outcome = json.loads(feedback_result.outcome_summary_path.read_text(encoding="utf-8"))
        return ReviewConsensusSnapshot(
            verdict=str(scorecard.get("verdict") or "unknown"),
            overall_score=float(scorecard.get("overall_score") or 0.0),
            return_warning_recommended=bool(scorecard.get("return_warning_recommended")),
            top_positive_aspects=tuple(str(item) for item in scorecard.get("top_positive_aspects") or []),
            top_negative_aspects=tuple(str(item) for item in scorecard.get("top_negative_aspects") or []),
            top_return_risk_aspects=tuple(str(item) for item in scorecard.get("top_return_risk_aspects") or []),
            feedback_count=int(outcome.get("feedback_count") or 0),
        )

    def _resolve_backend(self) -> ReviewRetrievalBackend:
        backend_name = self.config.retrieval_backend
        if backend_name == "auto":
            backend_name = "manifest" if self.config.manifest_path else "web_search"
        if backend_name == "manifest":
            if not self.config.manifest_path:
                raise ValueError("Manifest review retrieval requires `manifest_path`.")
            return ManifestReviewRetrievalBackend(self.config.manifest_path)
        if backend_name == "web_search":
            return WebSearchReviewRetrievalBackend(
                user_agent=self.config.retrieval_user_agent,
                timeout_seconds=self.config.retrieval_timeout_seconds,
            )
        raise ValueError(f"Unsupported review retrieval backend: {backend_name}")

    def _materialize_payload(self, document: DiscoveredReviewDocument) -> tuple[str, str, str]:
        if document.source_path:
            path = Path(document.source_path)
            return path.read_text(encoding="utf-8"), str(path), path.suffix or ".txt"
        if document.url:
            payload = self._fetch_url_text(document.url)
            suffix = ".html" if _looks_like_html(document.url) else ".txt"
            return payload, document.url, suffix
        raise ValueError(f"Review document {document.title!r} has neither source_path nor url")

    def _fetch_url_text(self, url: str) -> str:
        request_specs = (
            {
                "User-Agent": self.config.retrieval_user_agent,
                "Accept": "text/html, text/plain;q=0.9, */*;q=0.1",
            },
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            },
        )
        last_error: Exception | None = None
        for headers in request_specs:
            try:
                request = Request(url, headers=headers)
                with urlopen(request, timeout=self.config.retrieval_timeout_seconds) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    return response.read().decode(charset, errors="replace")
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                continue
        if last_error is None:
            raise RuntimeError(f"Unable to fetch {url}")
        raise last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve review pages and run BAFFLE product-feedback analysis.")
    parser.add_argument(
        "--query",
        default="",
        help="Review-retrieval query. If omitted, a local dashboard will open to collect it.",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--retrieval-backend", choices=["auto", "manifest", "web_search"], default="auto")
    parser.add_argument("--target-docs", type=int, default=5)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--consensus-batch-size", type=int, default=1)
    parser.add_argument("--consensus-score-tolerance", type=float, default=0.03)
    parser.add_argument("--consensus-stable-passes", type=int, default=1)
    parser.add_argument("--disable-consensus", action="store_true")
    parser.add_argument("--product-name", default="")
    parser.add_argument("--brand-name", default="")
    parser.add_argument("--analysis-question", default="")
    parser.add_argument("--discovery-only", action="store_true")
    args = parser.parse_args()
    query = _resolve_query_for_main(args)

    runner = ProductFeedbackQueryAgenticRunner(
        ProductFeedbackQueryAgenticConfig(
            query=query,
            outdir=Path(args.outdir),
            manifest_path=Path(args.manifest) if args.manifest else None,
            retrieval_backend=args.retrieval_backend,
            target_documents=args.target_docs,
            max_documents=args.max_docs,
            consensus_enabled=not args.disable_consensus,
            consensus_batch_size=args.consensus_batch_size,
            consensus_score_tolerance=args.consensus_score_tolerance,
            consensus_required_stable_passes=args.consensus_stable_passes,
            product_name=args.product_name,
            brand_name=args.brand_name,
            analysis_question=args.analysis_question,
            discovery_only=args.discovery_only,
        )
    )
    result = runner.run()
    if result.corpus_synthesis_result is not None:
        _open_dashboard_artifact(result.corpus_synthesis_result.dashboard_path)
    elif result.product_feedback_result is not None:
        _open_dashboard_artifact(result.product_feedback_result.dashboard_path)
    print(f"[BAFFLE product_feedback_query_agentic] selected reviews: {len(result.selected_documents)}")
    print(f"[BAFFLE product_feedback_query_agentic] materialized feedback manifest: {result.materialized_feedback_manifest_path}")
    if result.product_feedback_result is not None:
        print(f"[BAFFLE product_feedback_query_agentic] report: {result.product_feedback_result.report_path}")
        print(f"[BAFFLE product_feedback_query_agentic] dashboard: {result.product_feedback_result.dashboard_path}")
    if result.corpus_synthesis_result is not None:
        print(f"[BAFFLE product_feedback_query_agentic] corpus synthesis: {result.corpus_synthesis_result.dashboard_path}")


if __name__ == "__main__":
    main()
