"""Live integration tests for Overton API.

These tests execute against the real Overton API when an API key is present
in the environment.
"""

import os

import pandas as pd
import pytest

from discovery_utils.getters.overton import OvertonAuthError
from discovery_utils.getters.overton import OvertonGetter


pytestmark = pytest.mark.integration


def _have_key() -> bool:
    """Return True if OVERTON_API_KEY is configured in the environment."""
    return bool(os.getenv("OVERTON_API_KEY"))


@pytest.mark.skipif(not _have_key(), reason="Set OVERTON_API_KEY to run live Overton tests")
def test_live_search_minimal():
    """Perform a small live search and validate basic schema."""
    og = OvertonGetter()
    df = og.search_documents(query="climate", max_results=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 1
    if len(df):
        assert {"id", "title", "content"}.issubset(df.columns)


@pytest.mark.skipif(not _have_key(), reason="Set OVERTON_API_KEY to run live Overton tests")
def test_live_facets_smoke():
    """Request facets for a simple query and ensure a dict is returned."""
    og = OvertonGetter()
    facets = og.get_facets("climate")
    assert isinstance(facets, dict)


@pytest.mark.skipif(not _have_key(), reason="Set OVERTON_API_KEY to run live Overton tests")
def test_live_generate_id_set_permissions():
    """Attempt set generation; mark expected unauthorised keys as xfail."""
    og = OvertonGetter()
    try:
        sid = og.generate_id_set(["10.1038/nature12345"], identifier_type="dois")
        assert sid.startswith("set:")
    except OvertonAuthError:
        pytest.xfail("API key not entitled for generate_id_set on this account")


@pytest.mark.skipif(not _have_key(), reason="Set OVERTON_API_KEY to run live Overton tests")
def test_live_source_filters_smoke():
    """Source filters should be accepted and yield a small result set."""
    og = OvertonGetter()
    df = og.search_documents(query="energy", source_type="government", max_results=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 1


@pytest.mark.skipif(not _have_key(), reason="Set OVERTON_API_KEY to run live Overton tests")
def test_live_excluding_source_smoke():
    """excluding_source should be accepted without server errors."""
    og = OvertonGetter()
    df = og.search_documents(query="health", excluding_source="who", max_results=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 1


@pytest.mark.skipif(not _have_key(), reason="Set OVERTON_API_KEY to run live Overton tests")
def test_live_id_set_roundtrip():
    """If set generation is permitted, a subsequent set-based search should succeed."""
    og = OvertonGetter()
    try:
        sid = og.generate_id_set(["10.1038/nature12345"], identifier_type="dois")
    except OvertonAuthError:
        pytest.xfail("API key not entitled for generate_id_set on this account")
        return
    df = og.search_with_id_set(sid, field="plain_dois_cited", max_results=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 1
