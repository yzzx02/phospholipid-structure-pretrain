import io
import json
import urllib.error

import pytest

from structure_pretrain.pubchem_polar.client import PubChemClient, PubChemError, run_query


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


def test_large_logical_batch_uses_short_cid_urls(tmp_path):
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        cid_text = request.full_url.split("/cid/")[1].split("/")[0]
        cids = [int(c) for c in cid_text.split(",")]
        assert len(cids) <= 150
        return Response({"PropertyTable": {"Properties": [{"CID": c} for c in cids]}})

    client = PubChemClient(tmp_path, opener=opener, sleeper=lambda _: None)
    assert len(client.get_properties(list(range(1, 501)))) == 500
    assert len(calls) == 4
    assert len(client.get_properties(list(range(1, 501)))) == 500
    assert len(calls) == 4
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
