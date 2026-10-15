import hashlib
import json
import socket
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PROPERTIES = "ConnectivitySMILES,SMILES,MolecularFormula,MolecularWeight,Charge,InChIKey,Title"


class PubChemError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f"PubChem {status}: {message}")
        self.status = status


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class PubChemClient:
    def __init__(
        self,
        cache_dir,
        requests_per_second=3,
        max_retries=6,
        opener=None,
        sleeper=time.sleep,
        clock=time.monotonic,
    ):
        if not 0 < requests_per_second <= 4:
            raise ValueError("requests_per_second must be >0 and <=4")
        self.cache_dir = Path(cache_dir)
        (self.cache_dir / "responses").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.cache_dir / "properties.sqlite", timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS properties (cid INTEGER PRIMARY KEY, data TEXT)"
        )
        self.interval = 1 / requests_per_second
        self.max_retries = max_retries
        self.opener = opener or urllib.request.urlopen
        self.sleep, self.clock = sleeper, clock
        self.last_request = -float("inf")
        self.stats = {"network_requests": 0, "response_cache_hits": 0, "retries": 0}

    def get(self, path, params=None):
        url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
        key = hashlib.sha256(url.encode()).hexdigest()
        cached = self.cache_dir / "responses" / f"{key}.json"
        if cached.exists():
            self.stats["response_cache_hits"] += 1
            return json.loads(cached.read_text(encoding="utf-8"))["data"]
        for attempt in range(self.max_retries + 1):
            self.sleep(max(0, self.interval - (self.clock() - self.last_request)))
            self.last_request = self.clock()
            self.stats["network_requests"] += 1
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "phospholipid-structure-pretrain/0.1 (polar-review)",
                    "Accept": "application/json",
                },
            )
            try:
                with self.opener(request, timeout=45) as response:
                    data = json.loads(response.read())
                    throttle = response.headers.get("X-Throttling-Control", "")
                if "Fault" in data:
                    raise PubChemError(400, data["Fault"])
                atomic_json(
                    cached,
                    {
                        "url": url,
                        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "data": data,
                    },
                )
                if "Red" in throttle:
                    self.sleep(10)
                return data
            except urllib.error.HTTPError as exc:
                message = exc.read().decode(errors="replace")[:1500]
                if exc.code == 404 and '"PUGREST.NotFound"' in message:
                    # Negative search result is cacheable, distinct from network failure.
                    data = {"IdentifierList": {"CID": []}}
                    atomic_json(cached, {"url": url, "data": data})
                    return data
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise PubChemError(exc.code, message) from exc
                retry_after = exc.headers.get("Retry-After", "0") if exc.headers else "0"
                delay = max(2**attempt, float(retry_after) if retry_after.isdigit() else 0)
                error = PubChemError(exc.code, message)
            except (
                urllib.error.URLError,
                TimeoutError,
                socket.timeout,
                ConnectionError,
                OSError,
            ) as exc:
                delay, error = 2**attempt, PubChemError("network", str(exc))
            if attempt == self.max_retries:
                raise error
            self.stats["retries"] += 1
            self.sleep(min(delay, 60))
        raise AssertionError("unreachable")

    def query(self, smarts, cap):
        data = self.get(
            "/compound/fastsubstructure/smarts/cids/JSON", {"smarts": smarts, "MaxRecords": cap}
        )
        if "CID" not in data.get("IdentifierList", {}):
            raise PubChemError("unexpected_response", "No CID list; completeness unknown")
        cids = data["IdentifierList"]["CID"]
        if not isinstance(cids, list) or any(type(c) is not int or c <= 0 for c in cids):
            raise PubChemError("unexpected_response", "Invalid CID list")
        return sorted(set(cids))

    def get_properties(self, cids):
        if not cids:
            return []
        placeholders = ",".join("?" for _ in cids)
        existing = {
            r[0]: json.loads(r[1])
            for r in self.db.execute(
                f"SELECT cid,data FROM properties WHERE cid IN ({placeholders})", cids
            )
        }
        missing = sorted(set(cids) - existing.keys())
        # PubChem's front-end can reject very long CID URLs with HTTP 403.
        # Keep each GET short; logical output batches may still be larger.
        if len(missing) > 150:
            for start in range(0, len(missing), 150):
                existing.update(
                    {row["CID"]: row for row in self.get_properties(missing[start : start + 150])}
                )
        elif missing:
            data = self.get(
                f"/compound/cid/{','.join(map(str, missing))}/property/{PROPERTIES}/JSON"
            )
            rows = data.get("PropertyTable", {}).get("Properties", [])
            with self.db:
                for row in rows:
                    cid = row["CID"]
                    self.db.execute(
                        "INSERT OR REPLACE INTO properties VALUES (?,?)", (cid, json.dumps(row))
                    )
                    existing[cid] = row
        unavailable = sorted(set(cids) - existing.keys())
        if unavailable:
            raise PubChemError("properties_missing", unavailable)
        return [existing[cid] for cid in cids]

    def properties_by_name(self, name):
        data = self.get(
            f"/compound/name/{urllib.parse.quote(name, safe='')}/property/{PROPERTIES}/JSON"
        )
        rows = data.get("PropertyTable", {}).get("Properties", [])
        with self.db:
            for row in rows:
                self.db.execute(
                    "INSERT OR REPLACE INTO properties VALUES (?,?)", (row["CID"], json.dumps(row))
                )
        return rows

    def close(self):
        self.db.close()


def run_query(client, query, cap, full=False):
    """Underfilled MaxRecords is complete; saturated/failed searches are never complete."""
    result = {
        "query_id": query["query_id"],
        "family": query["family"],
        "smarts": query["smarts"],
        "requested_max_records": cap,
        "query_sha256": hashlib.sha256(query["smarts"].encode()).hexdigest(),
        "splits": [],
    }
    cids, complete, error = [], False, None
    try:
        cids = client.query(query["smarts"], cap)
        complete = len(cids) < cap
    except PubChemError as exc:
        error = str(exc)
    if full and not complete:
        branches = query.get("split_smarts", [])
        split_complete = bool(branches)
        for smarts in branches:
            try:
                subset = client.query(smarts, cap)
                done = len(subset) < cap
                cids = sorted(set(cids) | set(subset))
                split = {
                    "smarts": smarts,
                    "count": len(subset),
                    "complete": done,
                    "status": "COMPLETE" if done else "QUERY_TRUNCATED",
                }
            except PubChemError as exc:
                done = False
                split = {
                    "smarts": smarts,
                    "count": 0,
                    "complete": False,
                    "status": "QUERY_FAILED",
                    "error": str(exc),
                }
            result["splits"].append(split)
            split_complete &= done
        complete = split_complete
    result.update(
        {
            "cid_count": len(cids),
            "complete": complete,
            "status": "COMPLETE"
            if complete
            else "QUERY_FAILED"
            if error and not cids
            else "QUERY_TRUNCATED"
            if full
            else "SAMPLED_PARTIAL",
        }
    )
    if error:
        result["initial_error"] = error
    return result, cids
