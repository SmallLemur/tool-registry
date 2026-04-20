"""Unit tests for pure-logic helpers in registry_manager.

These helpers are at risk of drift during Phase 1 (embedding service merge);
tests pin their current behavior so the shared-service refactor doesn't
silently change what lands in Milvus.

Registry_manager's module-level imports pull pymilvus + full config chain,
which can't build on Python 3.14 yet. Skip cleanly when those deps are
missing; CI with Python 3.12 will run them.
"""

import pytest

# Skip the whole module if the import chain isn't available (pymilvus, etc.)
pytest.importorskip("pymilvus", reason="tool-registry needs pymilvus for full imports")

from datetime import datetime, timezone, timedelta

from app.core.registry_manager import (
    ServiceHealth,
    _build_search_text,
    _capability_id,
    _fingerprint,
)


# ── Fingerprint ───────────────────────────────────────────────

def test_fingerprint_is_deterministic():
    m = {"name": "calendar", "actions": [{"name": "list"}, {"name": "create"}]}
    assert _fingerprint(m) == _fingerprint(m)


def test_fingerprint_stable_under_key_reordering():
    """Canonical JSON sorts keys — reordering the manifest must not change
    the fingerprint (otherwise every client would force a re-embed)."""
    a = {"name": "x", "version": "1", "actions": []}
    b = {"actions": [], "version": "1", "name": "x"}
    assert _fingerprint(a) == _fingerprint(b)


def test_fingerprint_changes_on_content_change():
    a = {"name": "x", "version": "1"}
    b = {"name": "x", "version": "2"}
    assert _fingerprint(a) != _fingerprint(b)


# ── Capability IDs ────────────────────────────────────────────

def test_capability_id_is_deterministic():
    assert _capability_id("s", "a") == _capability_id("s", "a")


def test_capability_id_is_unique_per_pair():
    assert _capability_id("s", "a") != _capability_id("s", "b")
    assert _capability_id("s1", "a") != _capability_id("s2", "a")


def test_capability_id_is_fixed_length_32():
    assert len(_capability_id("anything", "at all")) == 32


# ── Search text builder ───────────────────────────────────────

def test_build_search_text_includes_service_and_action():
    text = _build_search_text(
        service_name="calendar",
        service_description="Scheduled reminders",
        action={"name": "create_entry", "description": "Create an entry"},
    )
    assert "Service: calendar" in text
    assert "Scheduled reminders" in text
    assert "Action: create_entry" in text
    assert "Create an entry" in text


def test_build_search_text_lists_parameter_names():
    text = _build_search_text(
        service_name="calendar",
        service_description="desc",
        action={
            "name": "create_entry",
            "description": "d",
            "input_schema": {"properties": {"title": {}, "when": {}}},
        },
    )
    assert "Parameters:" in text
    assert "title" in text
    assert "when" in text


def test_build_search_text_omits_parameters_line_when_none():
    text = _build_search_text(
        service_name="clock",
        service_description="d",
        action={"name": "now", "description": "current time"},
    )
    assert "Parameters:" not in text


# ── ServiceHealth ─────────────────────────────────────────────

def test_service_health_fresh_is_healthy():
    h = ServiceHealth(
        service_name="x",
        version="1",
        fingerprint="abc",
        last_seen=datetime.now(timezone.utc),
        capability_count=3,
    )
    assert h.is_healthy(timeout_s=60) is True


def test_service_health_stale_is_unhealthy():
    h = ServiceHealth(
        service_name="x",
        version="1",
        fingerprint="abc",
        last_seen=datetime.now(timezone.utc) - timedelta(minutes=10),
        capability_count=3,
    )
    assert h.is_healthy(timeout_s=60) is False
