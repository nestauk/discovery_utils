"""Unit test suite for the Overton getter.

These tests mock API interactions to testbehaviour and edge cases.
"""

import os

from typing import Any
from typing import Dict
from typing import List

import pandas as pd
import pytest
import requests

from discovery_utils.getters.overton import OvertonAuthError
from discovery_utils.getters.overton import OvertonGetter
from discovery_utils.getters.overton import OvertonValidationError


@pytest.fixture(autouse=True)
def set_api_key_env(monkeypatch: pytest.MonkeyPatch):
    """Ensure an API key is present for constructor initialisation in unit tests."""
    monkeypatch.setenv("OVERTON_API_KEY", "test_api_key")


@pytest.fixture(autouse=True)
def no_rate_limit(monkeypatch: pytest.MonkeyPatch):
    """Disable rate limiting to keep unit tests fast and deterministic."""
    # Make rate limiter a no-op for fast tests
    monkeypatch.setattr(
        "discovery_utils.getters.overton.OvertonRateLimiter.wait_if_needed",
        lambda self: None,
    )


def make_docs(start_id: int, count: int) -> List[Dict[str, Any]]:
    docs = []
    for i in range(start_id, start_id + count):
        docs.append(
            {
                "policy_document_id": f"doc-{i}",
                "title": f"Title {i}",
                "snippet": f"Snippet {i}",
                "source": {"title": "Org", "country": "UK", "type": "government", "region": ["Europe"]},
                "cites": {"scholarly": [], "policy": [], "news": [], "people": []},
            }
        )
    return docs


def test_pagination_collects_multiple_pages(monkeypatch: pytest.MonkeyPatch):
    """Ensure pagination fetches multiple pages and tracks page sequence."""
    calls = {"pages": []}

    def fake_make_request(
        self, endpoint: str, params: Dict[str, Any], method: str = "GET", validate_response: bool = True
    ):
        assert endpoint == "documents.php"
        page = params.get("page", 1)
        calls["pages"].append(page)
        if page == 1:
            return {"results": make_docs(1, 50), "query": {"next_page_url": "next"}}
        if page == 2:
            return {"results": make_docs(51, 50), "query": {"next_page_url": "next"}}
        # stop after 2 pages
        return {"results": [], "query": {}}

    monkeypatch.setattr(OvertonGetter, "_make_request", fake_make_request)

    getter = OvertonGetter()
    df = getter.search_documents(query="x", max_results=120)
    # Expect to collect 100 from two pages
    assert len(df) == 100
    assert calls["pages"] == [1, 2, 3]  # third call discovers no more results


def test_pagination_respects_max_results(monkeypatch: pytest.MonkeyPatch):
    """Verify pagination stops once max_results is reached across pages."""

    def fake_make_request(
        self, endpoint: str, params: Dict[str, Any], method: str = "GET", validate_response: bool = True
    ):
        page = params.get("page", 1)
        if page == 1:
            return {"results": make_docs(1, 50), "query": {"next_page_url": "next"}}
        if page == 2:
            return {"results": make_docs(51, 50), "query": {"next_page_url": "next"}}
        return {"results": make_docs(101, 50), "query": {}}

    monkeypatch.setattr(OvertonGetter, "_make_request", fake_make_request)
    getter = OvertonGetter()
    df = getter.search_documents(query="x", max_results=120)
    assert len(df) == 120


def test_get_facets_includes_flag_and_returns_data(monkeypatch: pytest.MonkeyPatch):
    """Confirm show_search_facets flag is sent and facet results are cached."""
    calls = {"params": []}

    def fake_make_request(
        self, endpoint: str, params: Dict[str, Any], method: str = "GET", validate_response: bool = True
    ):
        calls["params"].append(params)
        assert endpoint == "documents.php"
        assert params.get("show_search_facets") in (True, "true")
        return {"facets": {"policy_source_country": [{"key": "UK", "doc_count": 10}]}}

    monkeypatch.setattr(OvertonGetter, "_make_request", fake_make_request)
    getter = OvertonGetter()
    # Populate general cache (no query)
    facets = getter.get_facets()
    assert "policy_source_country" in facets
    assert facets["policy_source_country"][0]["key"] == "UK"
    # cached access without query should not call again
    monkeypatch.setattr(OvertonGetter, "_make_request", lambda *args, **kwargs: pytest.fail("should use cache"))
    cached = getter.get_facets()
    assert cached["policy_source_country"][0]["key"] == "UK"


def test_generate_id_set_success(monkeypatch: pytest.MonkeyPatch):
    """Check successful ID set generation posts newline-separated identifiers."""
    # Build a dummy session that captures POST data
    class DummyResponse:
        def __init__(self, status_code=200, payload=None, headers=None, reason="OK"):
            self.status_code = status_code
            self._payload = payload or {}
            self.headers = headers or {}
            self.reason = reason

        def json(self):
            return self._payload

    class DummySession:
        def __init__(self):
            self.headers = {}
            self.timeout = 5
            self.last_post = None

        def get(self, url, params=None):
            return DummyResponse(200, {"results": [], "query": {}})

        def post(self, url, data=None):
            self.last_post = {"url": url, "data": data}
            return DummyResponse(200, {"set": "set:1:abc"})

    getter = OvertonGetter()
    getter.session = DummySession()
    set_id = getter.generate_id_set(["10.1/a", "10.2/b"], identifier_type="dois")
    assert set_id.startswith("set:1:")
    posted = getter.session.last_post
    assert posted is not None
    # multi-line form field
    assert posted["data"]["dois"] == "10.1/a\n10.2/b"


def test_generate_id_set_unauthorised_raises(monkeypatch: pytest.MonkeyPatch):
    """Assert 401 unauthorised during set generation raises OvertonAuthError."""

    class DummyResponse:
        def __init__(self, status_code, payload=None, headers=None, reason=""):
            self.status_code = status_code
            self._payload = payload or {}
            self.headers = headers or {}
            self.reason = reason

        def json(self):
            return self._payload

    class DummySession:
        def __init__(self):
            self.headers = {}
            self.timeout = 5

        def get(self, url, params=None):
            return DummyResponse(200, {"results": [], "query": {}})

        def post(self, url, data=None):
            return DummyResponse(401, {})

    getter = OvertonGetter()
    getter.session = DummySession()
    with pytest.raises(OvertonAuthError):
        getter.generate_id_set(["10.1/a"])  # default identifier_type="dois"


def test_build_params_country_mappings(monkeypatch: pytest.MonkeyPatch):
    """Validate special country mapping uses source_region not source_country."""
    # Intercept the first request and examine computed params
    captured = {}

    def fake_make_request(
        self, endpoint: str, params: Dict[str, Any], method: str = "GET", validate_response: bool = True
    ):
        captured.update(params)
        return {"results": [], "query": {}}

    monkeypatch.setattr(OvertonGetter, "_make_request", fake_make_request)
    getter = OvertonGetter()
    getter.search_documents(query="q", source_country="All but UK", max_results=10)
    # Should use source_region mapping rather than source_country
    assert "source_country" not in captured
    assert captured.get("source_region") == OvertonGetter.REGION_MAPPINGS["All but UK"]


def test_recent_documents_property_caches(monkeypatch: pytest.MonkeyPatch):
    """Ensure recent_documents caches the first computed DataFrame."""
    calls = {"count": 0}

    def fake_search(self, **kwargs):
        calls["count"] += 1
        return pd.DataFrame(
            [
                {
                    "id": "x",
                    "title": "t",
                    "content": "c",
                    "authors": [],
                    "topics": [],
                    "classifications": [],
                    "source_tags": [],
                    "other_identifiers": [],
                    "languages": [],
                    "highlights": [],
                    "source_region": [],
                    "cites_scholarly": [],
                    "cites_policy": [],
                    "cites_news": [],
                    "cites_people": [],
                    "abstract": "a",
                    "publication_year": "2024",
                    "venue": "v",
                    "source_country": "UK",
                    "source_type": "government",
                    "citation_count": 0,
                    "es_score": 0.0,
                }
            ]
        )

    monkeypatch.setattr(OvertonGetter, "search_documents", fake_search)
    getter = OvertonGetter()
    _ = getter.recent_documents
    _ = getter.recent_documents
    assert calls["count"] == 1


def test_search_with_id_set_passes_field_and_value(monkeypatch: pytest.MonkeyPatch):
    """Verify search_with_id_set forwards field, set ID, and ancillary params."""
    captured = {}

    def fake_search(self, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(OvertonGetter, "search_documents", fake_search)
    getter = OvertonGetter()
    getter.search_with_id_set("set:1:abc", field="plain_dois_cited", max_results=123, query="q")
    assert captured["plain_dois_cited"] == "set:1:abc"
    assert captured["max_results"] == 123
    assert captured["query"] == "q"


def test_format_for_screening_success_and_missing_cols():
    """Check screening format for valid input and error on missing required cols."""
    getter = OvertonGetter()
    df_ok = pd.DataFrame(
        [
            {"id": "1", "title": "T", "content": "C"},
            {"id": "2", "title": "U", "content": "V"},
        ]
    )
    res = getter.format_for_screening(df_ok)
    assert res["1"]["title"] == "T"
    assert res["2"]["content"] == "V"

    df_bad = pd.DataFrame([{"title": "T", "content": "C"}])
    with pytest.raises(OvertonValidationError):
        getter.format_for_screening(df_bad)


def test_truncate_and_safe_conversions():
    """Exercise truncate and safe numeric conversions for edge values."""
    getter = OvertonGetter()
    long = "x" * 1200
    truncated = getter._truncate_content(long, max_length=1000)
    assert truncated.endswith("...")
    assert len(truncated) <= 1003

    assert getter._truncate_content("") == "No content available"
    assert getter._safe_int_conversion(None) == 0
    assert getter._safe_int_conversion("10") == 10
    assert getter._safe_int_conversion("10.4") == 10
    assert getter._safe_int_conversion("bad") == 0

    assert getter._safe_float_conversion(None) == 0.0
    assert getter._safe_float_conversion("1.5") == 1.5
    assert getter._safe_float_conversion("bad") == 0.0


def test_invalid_params_raise():
    """Ensure invalid search parameters raise validation errors."""
    getter = OvertonGetter()
    with pytest.raises(OvertonValidationError):
        getter.search_documents(max_results=0)
    with pytest.raises(OvertonValidationError):
        getter.search_documents(page=0)
    with pytest.raises(OvertonValidationError):
        getter.search_documents(sort="not-a-valid-sort")
    with pytest.raises(OvertonValidationError):
        getter.search_documents(has_references=2)


def test_search_multiple_values_combines_and_dedupes(monkeypatch: pytest.MonkeyPatch):
    """Confirm multi-value search merges and de-duplicates results across calls."""
    # Two searches yield overlapping IDs, ensure dedupe and limit
    def fake_search(self, **kwargs):
        # simulate different subsets based on injected params
        if kwargs.get("source_country") == "UK":
            return pd.DataFrame(
                [
                    {"id": "1", "title": "t1", "content": "c1"},
                    {"id": "2", "title": "t2", "content": "c2"},
                ]
            )
        return pd.DataFrame(
            [
                {"id": "2", "title": "t2", "content": "c2"},
                {"id": "3", "title": "t3", "content": "c3"},
            ]
        )

    monkeypatch.setattr(OvertonGetter, "search_documents", fake_search)
    getter = OvertonGetter()
    df = getter.search_multiple_values({"source_country": ["UK", "USA"]}, max_results=10, query="q")
    assert sorted(df["id"].tolist()) == ["1", "2", "3"]


def test_retry_on_429_respects_retry_after_and_backoff(monkeypatch: pytest.MonkeyPatch):
    """Simulate 429 then success and verify backoff sleep duration is applied."""
    sleeps: List[float] = []

    def fake_sleep(x: float):
        sleeps.append(x)

    monkeypatch.setattr("time.sleep", lambda x: fake_sleep(x))
    monkeypatch.setattr(
        "discovery_utils.getters.overton.OvertonRateLimiter.wait_if_needed",
        lambda self: None,
    )

    class DummyResponse:
        def __init__(self, status_code=200, payload=None, headers=None, reason="OK"):
            self.status_code = status_code
            self._payload = payload or {}
            self.headers = headers or {}
            self.reason = reason

        def json(self):
            return self._payload

    class DummySession:
        def __init__(self):
            self.headers = {}
            self.timeout = 5
            self.calls = 0

        def get(self, url, params=None):
            self.calls += 1
            if self.calls == 1:
                return DummyResponse(429, {}, {"Retry-After": "5"})
            return DummyResponse(200, {"results": [], "query": {}})

    getter = OvertonGetter(backoff_factor=2.0)
    getter.session = DummySession()
    # single request, first returns 429 then success; ensure retry occurred and sleep used min(5, 1*2.0)=2.0
    getter.search_documents(query="x", max_results=10)
    assert 2.0 in sleeps


def test_connection_error_then_success(monkeypatch: pytest.MonkeyPatch):
    """Simulate a connection error followed by success to test retry flow."""
    sleeps: List[float] = []
    monkeypatch.setattr("time.sleep", lambda x: sleeps.append(x))
    monkeypatch.setattr(
        "discovery_utils.getters.overton.OvertonRateLimiter.wait_if_needed",
        lambda self: None,
    )

    class DummyResponse:
        def __init__(self):
            self.status_code = 200
            self.reason = "OK"

        def json(self):
            return {"results": [], "query": {}}

    class DummySession:
        def __init__(self):
            self.headers = {}
            self.timeout = 5
            self.calls = 0

        def get(self, url, params=None):
            self.calls += 1
            if self.calls == 1:
                raise requests.ConnectionError("boom")
            return DummyResponse()

    getter = OvertonGetter(max_retries=1, backoff_factor=0.5)
    getter.session = DummySession()
    getter.search_documents(query="x", max_results=10)
    # one backoff sleep happened from the failed attempt
    assert sleeps and sleeps[0] == 0.5


def test_timeout_exhausts_and_breaks_pagination(monkeypatch: pytest.MonkeyPatch):
    """Simulate a timeout with zero retries and verify pagination returns empty."""
    monkeypatch.setattr(
        "discovery_utils.getters.overton.OvertonRateLimiter.wait_if_needed",
        lambda self: None,
    )

    class DummySession:
        def __init__(self):
            self.headers = {}
            self.timeout = 5

        def get(self, url, params=None):
            raise requests.Timeout("slow")

    getter = OvertonGetter(max_retries=0)
    getter.session = DummySession()
    # Pagination should catch and break, returning an empty DataFrame
    df = getter.search_documents(query="x", max_results=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_clear_cache_resets_everything(monkeypatch: pytest.MonkeyPatch):
    """Verify response and property caches are cleared by clear_cache()."""
    # enable response cache and property caches
    calls = {"get": 0}

    class DummyResponse:
        def __init__(self):
            self.status_code = 200
            self.reason = "OK"

        def json(self):
            return {"results": [], "query": {}}

    class DummySession:
        def __init__(self):
            self.headers = {}
            self.timeout = 5

        def get(self, url, params=None):
            calls["get"] += 1
            return DummyResponse()

    g = OvertonGetter(cache_enabled=True)
    g.session = DummySession()
    # Populate response cache via a request
    g.search_documents(query="y", max_results=10)
    # Populate property caches
    monkeypatch.setattr(OvertonGetter, "get_facets", lambda self: {"x": []})
    _ = g.facets
    _ = g.recent_documents  # will call search_documents, already mocked above
    # Clear caches
    g.clear_cache()
    # Next request should hit session.get again (cache cleared)
    g.search_documents(query="y", max_results=10)
    assert calls["get"] >= 2
    # Facets property should call get_facets again after clear
    state = {"called": False}
    monkeypatch.setattr(OvertonGetter, "get_facets", lambda self: state.__setitem__("called", True) or {"x": []})
    _ = g.facets
    assert state["called"] is True


def test_facets_property_ttl(monkeypatch: pytest.MonkeyPatch):
    """Check facets property respects TTL and avoids redundant refreshes."""
    g = OvertonGetter()
    g._facets_cache = {"a": []}
    g._facets_cache_time = 0  # very old
    called = {"count": 0}
    monkeypatch.setattr(
        OvertonGetter, "get_facets", lambda self: called.__setitem__("count", called["count"] + 1) or {"b": []}
    )
    _ = g.facets
    assert called["count"] == 1
    # Now within TTL should not call get_facets again
    g._facets_cache_time = 10**12  # far future compared to current time
    _ = g.facets
    assert called["count"] == 1


def test_pagination_stop_on_missing_next_page(monkeypatch: pytest.MonkeyPatch):
    """Stop paginating when next_page_url is absent in the API response."""

    def fake_make_request(
        self, endpoint: str, params: Dict[str, Any], method: str = "GET", validate_response: bool = True
    ):
        if params.get("page", 1) == 1:
            return {"results": make_docs(1, 50), "query": {}}  # no next_page_url present
        return {"results": make_docs(51, 50), "query": {}}

    monkeypatch.setattr(OvertonGetter, "_make_request", fake_make_request)
    g = OvertonGetter()
    df = g.search_documents(query="x", max_results=500)
    assert len(df) == 50


def test_semantic_search_params_and_cache_key_excludes_api_key():
    """Ensure semantic params are passed and cache keys omit API secrets."""
    g = OvertonGetter()
    params = g._build_search_params(query="q", semantic_search=True, min_similarity=0.7)
    assert params["squery"] == "q"
    assert params["min_similarity"] == 0.7
    key = g._generate_cache_key("documents.php", {**params, "api_key": "secret"})
    assert "secret" not in key


def test_process_documents_normalises_types(monkeypatch: pytest.MonkeyPatch):
    """Normalise string/list fields and citations into consistent list types."""
    raw_doc = {
        "policy_document_id": "p1",
        "title": "t",
        "authors": "single author",
        "topics": "energy",
        "classifications": "economics",
        "source_tags": "tag1",
        "other_identifiers": "x",
        "languages": "en",
        "highlights": "h",
        "source": {"title": "Org", "country": "UK", "type": "government", "region": "Europe"},
        "cites": {"scholarly": {}, "policy": {}, "news": {}, "people": {}},
        "snippet": "s",
    }

    def fake_make_request(
        self, endpoint: str, params: Dict[str, Any], method: str = "GET", validate_response: bool = True
    ):
        return {"results": [raw_doc], "query": {}}

    monkeypatch.setattr(OvertonGetter, "_make_request", fake_make_request)
    g = OvertonGetter()
    df = g.search_documents(query="q", max_results=10)
    row = df.iloc[0]
    # lists normalised
    assert isinstance(row["authors"], list)
    assert isinstance(row["topics"], list)
    assert isinstance(row["classifications"], list)
    assert isinstance(row["source_tags"], list)
    assert isinstance(row["other_identifiers"], list)
    assert isinstance(row["languages"], list)
    assert isinstance(row["highlights"], list)
    assert isinstance(row["source_region"], list)
    # citations normalised to lists
    assert isinstance(row["cites_scholarly"], list)
    assert isinstance(row["cites_policy"], list)
    assert isinstance(row["cites_news"], list)
    assert isinstance(row["cites_people"], list)


def test_generate_id_set_isbns_with_warnings(monkeypatch: pytest.MonkeyPatch):
    """Exercise ISBN set generation and tolerate warning presence in response."""

    class DummyResponse:
        def __init__(self, status_code=200, payload=None):
            self.status_code = status_code
            self._payload = payload or {}
            self.headers = {}
            self.reason = "OK"

        def json(self):
            return self._payload

    class DummySession:
        def __init__(self):
            self.headers = {}
            self.timeout = 5

        def get(self, url, params=None):
            return DummyResponse(200, {"results": [], "query": {}})

        def post(self, url, data=None):
            return DummyResponse(200, {"set": "set:1:isbn", "warnings": ["note"]})

    g = OvertonGetter()
    g.session = DummySession()
    set_id = g.generate_id_set(["9780262033848"], identifier_type="isbns")
    assert set_id.startswith("set:1:")
