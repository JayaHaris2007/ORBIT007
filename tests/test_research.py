"""Characterization tests for the research package: rate limiter, credential scrubbing, and normalization."""

from unittest.mock import MagicMock
import pytest
from fastapi import HTTPException
from research import config, rate_limit, search_engine


def make_mock_request(client_ip="127.0.0.1"):
    request = MagicMock()
    request.client.host = client_ip
    return request


def test_rate_limiter_allows_under_limit():
    rate_limit.reset()
    req = make_mock_request("192.168.1.100")
    # Up to RATE_LIMIT_REQUESTS should succeed without exception
    for _ in range(config.RATE_LIMIT_REQUESTS):
        rate_limit.enforce(req)


def test_rate_limiter_blocks_and_sets_retry_after():
    rate_limit.reset()
    req = make_mock_request("192.168.1.101")
    for _ in range(config.RATE_LIMIT_REQUESTS):
        rate_limit.enforce(req)

    # The (LIMIT + 1)-th request must raise HTTPException(429)
    with pytest.raises(HTTPException) as exc_info:
        rate_limit.enforce(req)

    assert exc_info.value.status_code == 429
    assert "Too many research requests" in exc_info.value.detail
    assert "Retry-After" in exc_info.value.headers
    retry_after = int(exc_info.value.headers["Retry-After"])
    assert retry_after >= 1


def test_rate_limiter_isolates_different_ips():
    rate_limit.reset()
    req_a = make_mock_request("10.0.0.1")
    req_b = make_mock_request("10.0.0.2")

    # Exhaust A's limit
    for _ in range(config.RATE_LIMIT_REQUESTS):
        rate_limit.enforce(req_a)

    with pytest.raises(HTTPException):
        rate_limit.enforce(req_a)

    # Client B should still be allowed without error
    rate_limit.enforce(req_b)


def test_credential_scrubbing():
    secret_key = "tvly-secret-key-xyz123456"
    raw_error_message = f"Error connecting to api.tavily.com with key={secret_key} timed out."

    scrubbed = search_engine._scrub(raw_error_message, secret_key)
    assert secret_key not in scrubbed
    assert "***" in scrubbed
    assert "key=*** timed out." in scrubbed

    # Empty secret should not alter the message
    assert search_engine._scrub("Normal error", "") == "Normal error"


def test_search_not_configured_when_key_missing(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(search_engine.SearchNotConfigured) as exc_info:
        search_engine.search("test query")
    assert "Web research is not configured" in str(exc_info.value)


def test_search_engine_normalization():
    payload = {
        "results": [
            {
                "url": "https://www.example.com/docs/intro#section1",
                "title": "Example Documentation",
                "content": "Introductory guide to python programming.",
                "score": 0.9523,
            },
            # Duplicate URL (different fragment/www)
            {
                "url": "http://example.com/docs/intro",
                "title": "Duplicate Link",
                "content": "Duplicate content",
                "score": 0.88,
            },
            # Invalid URL protocol
            {
                "url": "ftp://files.example.com/data",
                "title": "FTP Link",
                "content": "Should be ignored",
                "score": 0.5,
            },
        ]
    }

    sources = search_engine._normalize(payload, max_results=5)
    assert len(sources) == 1  # Deduplicated and invalid filtered
    first = sources[0]
    assert first["id"] == 1
    assert first["domain"] == "example.com"
    assert first["title"] == "Example Documentation"
    assert first["score"] == 0.9523
