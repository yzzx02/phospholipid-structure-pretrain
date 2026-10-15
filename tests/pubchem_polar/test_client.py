import io
import json
import urllib.error
import urllib.parse

import pytest

from structure_pretrain.pubchem_polar.client import PubChemClient, PubChemError, run_query
from structure_pretrain.pubchem_polar.pipeline import property_batches


class Response:
    headers = {}

    def __init__(self, data):
        self.data = data

    def read(self):
        return json.dumps(self.data).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_query_cache_persists_on_resume(tmp_path):
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        return Response({"IdentifierList": {"CID": [1, 2]}})

    client = PubChemClient(tmp_path, opener=opener)
    assert client.query("P", 10) == [1, 2]
    client.close()
    resumed = PubChemClient(tmp_path, opener=opener)
    assert resumed.query("P", 10) == [1, 2]
    assert len(calls) == 1
    resumed.close()


def test_properties_not_refetched_across_query_batches(tmp_path):
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        return Response({"PropertyTable": {"Properties": [{"CID": 1}, {"CID": 2}]}})

    client = PubChemClient(tmp_path, opener=opener)
    assert len(client.get_properties([1, 2])) == 2
    assert client.get_properties([2]) == [{"CID": 2}]
    assert len(calls) == 1
    client.close()


def test_large_logical_batch_uses_post_and_reuses_cache(tmp_path):
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        assert request.get_method() == "POST"
        assert len(request.full_url) < 250
        cid_text = urllib.parse.parse_qs(request.data.decode())["cid"][0]
        cids = [int(c) for c in cid_text.split(",")]
        assert len(cids) == 500
        return Response({"PropertyTable": {"Properties": [{"CID": c} for c in cids]}})

    client = PubChemClient(tmp_path, opener=opener, sleeper=lambda _: None)
    assert len(client.get_properties(list(range(1, 501)))) == 500
    assert len(calls) == 1
    assert len(client.get_properties(list(range(1, 501)))) == 500
    assert len(calls) == 1
    client.close()


def test_single_prefetch_worker_uses_real_sqlite_cache(tmp_path):
    def opener(request, timeout):
        cid = int(request.full_url.split("/cid/")[1].split("/")[0])
        return Response({"PropertyTable": {"Properties": [{"CID": cid}]}})

    client = PubChemClient(tmp_path, opener=opener, sleeper=lambda _: None)
    batches = list(property_batches(client, [1, 2], 1))
    assert all(error is None for _, _, _, error in batches)
    assert [rows[0]["CID"] for _, _, rows, _ in batches] == [1, 2]
    assert client.get_properties([1, 2]) == [{"CID": 1}, {"CID": 2}]
    assert client.stats["network_requests"] == 2
    client.close()


def test_429_503_retry_and_all_requests_rate_limited(tmp_path):
    current, sent, sleeps = [0.0], [], []

    def sleep(seconds):
        current[0] += seconds
        sleeps.append(seconds)

    def opener(request, timeout):
        sent.append(current[0])
        if len(sent) < 3:
            raise urllib.error.HTTPError(
                request.full_url, [429, 503][len(sent) - 1], "busy", {}, io.BytesIO(b"{}")
            )
        return Response({"IdentifierList": {"CID": [1]}})

    client = PubChemClient(
        tmp_path, requests_per_second=4, opener=opener, sleeper=sleep, clock=lambda: current[0]
    )
    assert client.query("P", 10) == [1]
    assert all(b - a >= 0.25 for a, b in zip(sent, sent[1:]))
    assert client.stats["retries"] == 2
    assert 1 in sleeps and 2 in sleeps
    client.close()


def test_truncated_full_query_splits_and_unions():
    class Client:
        def query(self, smarts, cap):
            return {"root": [1, 2], "a": [1], "b": [2]}[smarts]

    query = {"query_id": "q", "family": "phosphate", "smarts": "root", "split_smarts": ["a", "b"]}
    report, cids = run_query(Client(), query, 2, full=True)
    assert cids == [1, 2] and report["complete"] and len(report["splits"]) == 2


def test_saturated_leaf_is_never_called_complete():
    class Client:
        def query(self, smarts, cap):
            return [1, 2]

    query = {"query_id": "q", "family": "phosphate", "smarts": "P", "split_smarts": ["P"]}
    report, _ = run_query(Client(), query, 2, full=True)
    assert not report["complete"] and report["status"] == "QUERY_TRUNCATED"


def test_sample_is_labelled_partial():
    class Client:
        def query(self, smarts, cap):
            return [1, 2]

    report, _ = run_query(Client(), {"query_id": "q", "family": "f", "smarts": "P"}, 2)
    assert report["status"] == "SAMPLED_PARTIAL" and not report["complete"]


def test_pending_response_cannot_be_mistaken_for_complete_empty_query(tmp_path):
    client = PubChemClient(tmp_path, opener=lambda *a, **k: Response({"Waiting": {"ListKey": "x"}}))
    with pytest.raises(PubChemError, match="completeness unknown"):
        client.query("P", 1000)
    client.close()


def test_failed_query_splits_and_keeps_incomplete_flag():
    class Client:
        def query(self, smarts, cap):
            if smarts != "a":
                raise PubChemError(503, "busy")
            return [1]

    query = {"query_id": "q", "family": "f", "smarts": "root", "split_smarts": ["a", "b"]}
    report, cids = run_query(Client(), query, 10, full=True)
    assert cids == [1] and not report["complete"]
    assert report["splits"][1]["status"] == "QUERY_FAILED"


def test_persistent_transport_keeps_retries_under_client_control(tmp_path, monkeypatch):
    import requests

    created, calls, closed = [], [], []

    class Reply:
        headers = {}

        def __init__(self, status, data):
            self.status_code, self.content = status, json.dumps(data).encode()

        def close(self):
            pass

    class Session:
        def __init__(self):
            created.append(self)

        def mount(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            calls.append(kwargs)
            return (
                Reply(503, {})
                if len(calls) == 1
                else Reply(200, {"IdentifierList": {"CID": [1004]}})
            )

        def close(self):
            closed.append(True)

    monkeypatch.setattr(requests, "Session", Session)
    client = PubChemClient(tmp_path, sleeper=lambda _: None)
    assert client.query("P", 10) == [1004]
    assert len(created) == 1 and len(calls) == 2
    assert all(call["allow_redirects"] is False for call in calls)
    assert client.stats["retries"] == 1
    client.close()
    assert closed == [True]


def test_two_senders_share_rate_limiter_and_safe_property_cache(tmp_path, monkeypatch):
    import threading
    import time

    import requests

    barrier, lock = threading.Barrier(2), threading.Lock()
    starts, sessions = [], []

    class Reply:
        status_code, headers = 200, {}

        def __init__(self, cid):
            self.content = json.dumps({"PropertyTable": {"Properties": [{"CID": cid}]}}).encode()

        def close(self):
            pass

    class Session:
        def __init__(self):
            sessions.append(self)

        def mount(self, *args, **kwargs):
            pass

        def request(self, method, url, **kwargs):
            with lock:
                starts.append(time.monotonic())
                ordinal = len(starts)
            if ordinal <= 2:
                barrier.wait(timeout=5)
            return Reply(int(url.split("/cid/")[1].split("/")[0]))

        def close(self):
            pass

    monkeypatch.setattr(requests, "Session", Session)
    client = PubChemClient(tmp_path, requests_per_second=4)
    batches = list(property_batches(client, [1, 2, 3, 4], 1, workers=2))
    assert all(error is None for _, _, _, error in batches)
    assert [rows[0]["CID"] for _, _, rows, _ in batches] == [1, 2, 3, 4]
    assert len(sessions) == 2
    assert all(b - a >= 0.24 for a, b in zip(sorted(starts), sorted(starts)[1:]))
    assert client.get_properties([1, 2, 3, 4]) == [{"CID": c} for c in (1, 2, 3, 4)]
    assert client.stats["network_requests"] == 4
    client.close()


def test_ncbi_html_gateway_803_is_retried_not_cached_as_empty_query(tmp_path):
    calls = []

    def opener(request, timeout):
        calls.append(True)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                404,
                "gateway",
                {},
                io.BytesIO(b"<html><title>NCBI - WWW Error 803 Diagnostic</title></html>"),
            )
        return Response({"IdentifierList": {"CID": [1004]}})

    client = PubChemClient(tmp_path, opener=opener, sleeper=lambda _: None)
    assert client.query("P", 100) == [1004]
    assert client.stats["retries"] == 1 and len(calls) == 2
    client.close()


def test_cached_only_returns_partial_properties_without_network(tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline mode sent a network request")

    client = PubChemClient(tmp_path, opener=forbidden, cached_only=True)
    with client.db:
        client.db.execute("INSERT INTO properties VALUES (?,?)", (1, json.dumps({"CID": 1})))
    assert client.get_properties([1, 2]) == [{"CID": 1}]
    with pytest.raises(PubChemError, match="offline_cache_miss"):
        client.query("P", 10)
    assert client.stats["network_requests"] == 0
    client.close()


def test_exhausted_429_suspends_all_new_requests(tmp_path):
    calls = []

    def opener(request, timeout):
        calls.append(True)
        raise urllib.error.HTTPError(request.full_url, 429, "quota", {}, io.BytesIO(b"quota"))

    client = PubChemClient(tmp_path, opener=opener, max_retries=0, sleeper=lambda _: None)
    assert client.get_properties([1]) == []
    assert client.network_suspension["status"] == 429
    assert client.get_properties([2, 3]) == []
    assert len(calls) == client.stats["network_requests"] == 1
    client.close()
