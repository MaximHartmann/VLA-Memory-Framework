"""SessionStore: the stored moments in an embedded LanceDB database (one directory), behind the framework's MemoryStore
interface (__len__, retrieve, describe_hit, fewshot_content), so the VLM writer's gate and examples use it unchanged.

Directory layout (`path`):
    moments.lance/     one row per stored moment: id, a vector column per stored key (key_colour, key_dinov2, key_siglip),
                       the two pictures (PNG), the payload
    usage.lance/       the use counters: id, times_retrieved, last_retrieval (rewritten at every episode end; kept apart
                       from the moments so that counting a retrieval never rewrites the pictures)
    info.json          the store's creation record and the number of updates (episodes written)
    retrievals.jsonl   one line per query: episode, step, key, filter, size, time, and the hits (id, similarity, session,
                       subgoal, provenance)

A new database stores the colour key, which also decides the near-duplicates of an episode (online.py), the retrieval key and
any further keys asked for; an existing one keeps the keys it was created with, and retrieves with any of them.

Retrieval: the k nearest moments by cosine similarity of the chosen key, among the rows the payload filter admits
(prefiltered, exact flat search), at most `per_episode` (2) from one past episode, never from `exclude_episode`: the rule of
the framework's ExemplarStore.retrieve. Eviction above `cap`: fifo removes the lowest ids (oldest insert) first;
use_recency removes the least retrieved first and among equally used the oldest.

Needs lancedb and pyarrow; the extension imports this module only when it builds the store.
"""
from __future__ import annotations

import io
import json
import pathlib
import threading
import time
from typing import Sequence

import lancedb
import numpy as np
import pyarrow as pa
from PIL import Image

from ...memory import MemoryStore
from ...writers.vlm.exemplar import ExemplarStore
from .keys import KeyEncoder
from .options import DEFAULT_EVICT, DEFAULT_FILTER, DEFAULT_KEY, EVICT, FILTERS, KEY_NAMES

PAYLOAD = (("session", pa.string()), ("task", pa.string()), ("instruction", pa.string()), ("plan", pa.string()),
           ("skill", pa.string()), ("subgoal", pa.int32()), ("phase", pa.int32()), ("labels", pa.string()),
           ("time", pa.float64()), ("episode", pa.string()), ("step", pa.int32()), ("outcome", pa.string()),
           ("skills_done", pa.int32()), ("provenance", pa.string()), ("labeller", pa.string()), ("update_no", pa.int32()))
USAGE_SCHEMA = pa.schema([pa.field("id", pa.int64()), pa.field("times_retrieved", pa.int64()),
                          pa.field("last_retrieval", pa.float64())])
HIT_COLUMNS = ["id", "episode", "step", "session", "subgoal", "provenance", "labels", "exterior", "wrist"]


def normalise_instruction(text) -> str:
    """The instruction as the filter compares it: lower case, single spaces, no final full stop."""
    return " ".join(str(text or "").lower().split()).rstrip(".")


def quote(text) -> str:
    """A string literal of the filter language."""
    return "'" + str(text).replace("'", "''") + "'"


def png_bytes(picture: np.ndarray) -> bytes:
    """The picture (its first three channels) as PNG bytes; the light compression keeps the inserts fast."""
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(picture, np.uint8)[..., :3]).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def png_array(data: bytes) -> np.ndarray:
    """PNG bytes as a uint8 array (H, W, 3)."""
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def moment_schema(dims: dict) -> pa.Schema:
    """The moments table: the id, a vector column per key (`dims`: key name -> length), the two pictures and the payload."""
    fields = [pa.field("id", pa.int64())]
    fields += [pa.field(f"key_{name}", pa.list_(pa.float32(), int(length))) for name, length in dims.items()]
    fields += [pa.field("exterior", pa.binary()), pa.field("wrist", pa.binary())]
    fields += [pa.field(name, kind) for name, kind in PAYLOAD]
    return pa.schema(fields)


def keys_of_new_database(key: str, keys: Sequence[str] | None) -> tuple[str, ...]:
    """The keys a new database stores, in KEY_NAMES order: the colour key, the retrieval key and the further `keys`."""
    wanted = {"colour", key, *(keys or ())}
    return tuple(name for name in KEY_NAMES if name in wanted)


class SessionStore(MemoryStore):
    """The database at `path` (created when missing). key: which key retrieves; filter: which payload field must equal
    the current episode's (none, task, instruction, plan, session); cap: most rows (0 = unbounded); evict: fifo or
    use_recency; keys: further keys a new database stores besides the colour and the retrieval key (an existing database
    keeps its own). The attribute `keys` lists the keys the database stores. The current episode's instruction, plan,
    session, episode and step are set with set_context()."""

    def __init__(self, path, task, key: str = DEFAULT_KEY, filter: str = DEFAULT_FILTER, cap: int = 0,
                 evict: str = DEFAULT_EVICT, session: str | None = None, keys: Sequence[str] | None = None,
                 encoder: KeyEncoder | None = None, device: str = "cpu", threads: int = 8, per_episode: int = 2,
                 log_queries: bool = True):
        if key not in KEY_NAMES or filter not in FILTERS or evict not in EVICT:
            raise ValueError(f"SessionStore: key in {KEY_NAMES}, filter in {FILTERS}, evict in {EVICT}; "
                             f"got {key}, {filter}, {evict}")
        if any(name not in KEY_NAMES for name in keys or ()):
            raise ValueError(f"SessionStore: keys in {KEY_NAMES}; got {list(keys)}")
        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.task, self.key, self.filter, self.cap, self.evict_rule = task, key, filter, int(cap or 0), evict
        self.session = session or time.strftime("session_%Y%m%d_%H%M%S")
        self.encoder = encoder or KeyEncoder(task, device=device, threads=threads)
        self.per_episode = int(per_episode)
        self.db = lancedb.connect(str(self.path))
        self.lock = threading.RLock()
        self.context = {"instruction": "", "plan": "", "session": self.session, "episode": None, "step": None}
        self.hits_cache: dict[int, dict] = {}      # the rows of the last queries' hits, for describe_hit and fewshot_content
        self.table, self.keys = self._open_or_create(keys)
        self.usage = self._read_usage()            # id -> [times retrieved, time of the last retrieval]
        self.count = self.table.count_rows()
        self.next_id = self._max_id() + 1
        self.info = self._read_info()
        self.log_file = open(self.path / "retrievals.jsonl", "a") if log_queries else None

    # ------------------------------------------------------------------ opening
    def _open_or_create(self, keys: Sequence[str] | None) -> tuple:
        """The moments table and the keys it stores: those of an existing table (which must include the retrieval key), or
        those of a new one."""
        if "moments" in self.db.list_tables().tables:
            table = self.db.open_table("moments")
            stored = tuple(name for name in KEY_NAMES if f"key_{name}" in table.schema.names)
            if self.key not in stored:
                raise ValueError(f"the database {self.path} stores the keys {list(stored)}, not {self.key!r}")
            return table, stored
        stored = keys_of_new_database(self.key, keys)
        dims = {name: self.encoder.dim(name) for name in stored}
        return self.db.create_table("moments", schema=moment_schema(dims)), stored

    def _read_usage(self) -> dict:
        """The use counters of the stored moments."""
        if "usage" not in self.db.list_tables().tables:
            return {}
        rows = self.db.open_table("usage").to_arrow().to_pydict()
        return {int(i): [int(n), float(t)] for i, n, t in zip(rows["id"], rows["times_retrieved"], rows["last_retrieval"])}

    def _max_id(self) -> int:
        """The highest id in the table (-1 for an empty one)."""
        if not self.count:
            return -1
        ids = self.scan(["id"])["id"].to_numpy()
        return int(ids.max()) if len(ids) else -1

    def _read_info(self) -> dict:
        """info.json, written when the database is created."""
        file = self.path / "info.json"
        if file.exists():
            return json.loads(file.read_text())
        info = {"format": "session_store v1", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "task": self.task.name,
                "updates": 0, "keys": list(self.keys), "dims": {name: self.encoder.dim(name) for name in self.keys}}
        self._write_info(info)
        return info

    def _write_info(self, info: dict):
        """Replace info.json in one step."""
        temporary = self.path / "info.json.tmp"
        temporary.write_text(json.dumps(info, indent=1))
        temporary.replace(self.path / "info.json")

    # ------------------------------------------------------------------ the query context
    def set_context(self, **fields):
        """instruction, plan (list of skills or text), session, episode, step of the current call."""
        if "instruction" in fields:
            fields["instruction"] = normalise_instruction(fields["instruction"])
        if isinstance(fields.get("plan"), (list, tuple)):
            fields["plan"] = " | ".join(fields["plan"])
        self.context.update(fields)

    def where_clause(self, exclude_episode=None) -> str | None:
        """The payload filter of the current episode, and the left-out episode."""
        parts = []
        if self.filter == "task":
            parts.append(f"task = {quote(self.task.name)}")
        elif self.filter in ("instruction", "plan", "session"):
            parts.append(f"{self.filter} = {quote(self.context[self.filter])}")
        if exclude_episode is not None:
            parts.append(f"episode != {quote(exclude_episode)}")
        return " AND ".join(parts) or None

    # ------------------------------------------------------------------ MemoryStore: reading
    def __len__(self) -> int:
        return int(self.count)

    def retrieve(self, images, k: int = 4, exclude_episode=None, per_episode: int | None = None):
        """[(id, similarity)] of the k nearest admitted moments (module docstring); counts the use of every hit."""
        with self.lock:
            started = time.perf_counter()
            query = self.encoder.encode(self.key, [list(images)])[0]
            encoded = time.perf_counter()
            hits, rows = self.search(query, k, self.where_clause(exclude_episode), per_episode or self.per_episode)
            now = time.time()
            for row in rows:
                self.hits_cache[int(row["id"])] = row
                use = self.usage.setdefault(int(row["id"]), [0, 0.0])
                use[0] += 1
                use[1] = now
            self._log_query(hits, rows, 1000 * (encoded - started), 1000 * (time.perf_counter() - encoded))
            return hits

    def search(self, query: np.ndarray, k: int, where: str | None, per_episode: int = 2, columns=None):
        """The k nearest moments of `query` by the retrieval key, at most `per_episode` of one episode: (hits [(id,
        similarity)], their rows as dicts). The database returns the `limit` nearest rows; when the per-episode cap leaves
        fewer than k of them, the search widens to four times as many rows, until it has k hits or has seen every row."""
        if not self.count:
            return [], []
        limit = min(max(8 * k, 48), self.count)
        while True:
            rows = self._nearest(query, limit, where, columns)
            hits, kept = self._capped_per_episode(rows, k, per_episode)
            if len(hits) >= k or len(rows) < limit or limit >= self.count:
                return hits, kept
            limit = min(limit * 4, self.count)

    def _nearest(self, query: np.ndarray, limit: int, where: str | None, columns=None) -> list[dict]:
        """The `limit` rows nearest to `query` by cosine distance of the retrieval key, nearest first, with `_distance`."""
        builder = self.table.search(np.asarray(query, np.float32), vector_column_name=f"key_{self.key}")
        builder = builder.distance_type("cosine").select(list(columns or HIT_COLUMNS) + ["_distance"]).limit(limit)
        if where:
            builder = builder.where(where, prefilter=True)
        return builder.to_arrow().to_pylist()

    @staticmethod
    def _capped_per_episode(rows: list[dict], k: int, per_episode: int) -> tuple[list, list]:
        """The first k rows with at most `per_episode` of the same episode: (hits [(id, similarity)], the rows)."""
        hits, kept, taken = [], [], {}
        for row in rows:
            if taken.get(row["episode"], 0) >= per_episode:
                continue
            taken[row["episode"]] = taken.get(row["episode"], 0) + 1
            hits.append((int(row["id"]), 1.0 - float(row["_distance"])))
            kept.append(row)
            if len(hits) >= k:
                break
        return hits, kept

    def _log_query(self, hits, rows, encode_ms, search_ms):
        """One line of retrievals.jsonl."""
        if self.log_file is None:
            return
        record = {"t": round(time.time(), 3), "episode": self.context.get("episode"), "step": self.context.get("step"),
                  "key": self.key, "filter": self.filter, "size": self.count, "encode_ms": round(encode_ms, 2),
                  "search_ms": round(search_ms, 2),
                  "hits": [[i, round(s, 4), r["session"], r["subgoal"], r["provenance"]] for (i, s), r in zip(hits, rows)]}
        self.log_file.write(json.dumps(record) + "\n")
        self.log_file.flush()

    def describe_hit(self, index) -> dict:
        """The episode, frame, skill index, session and provenance of a hit of the last queries."""
        row = self.hits_cache[int(index)]
        return {"episode": row["episode"], "frame": row["step"], "subgoal": row["subgoal"], "session": row["session"],
                "provenance": row["provenance"]}

    def fewshot_content(self, hits, skill_index, skill, encode_png):
        """The examples' text and pictures, written by the exemplar store from a store that holds just these hits."""
        rows = [self.hits_cache[int(index)] for index, _ in hits]
        examples = ExemplarStore(self.encoder.fingerprint, self.task)
        examples.images = [np.stack([png_array(row[camera]) for row in rows]) for camera in ("exterior", "wrist")]
        examples.meta = [dict(json.loads(row["labels"]), subgoal=row["subgoal"]) for row in rows]
        numbered = [(number, similarity) for number, (_, similarity) in enumerate(hits)]     # the entries of `examples`
        return examples.fewshot_content(numbered, skill_index, skill, encode_png)

    # ------------------------------------------------------------------ writing
    def insert(self, pictures: Sequence[Sequence[np.ndarray]], payloads: Sequence[dict], keys: dict | None = None) -> list:
        """Add moments: pictures [exterior, wrist] per moment, payload dicts (fields of PAYLOAD; labels may be a dict),
        keys {name: (n, D)} (a stored key that is missing is encoded here). Returns the new ids."""
        count = len(payloads)
        if not count:
            return []
        keys = dict(keys or {})
        for name in self.keys:
            if name not in keys:
                keys[name] = self.encoder.encode(name, pictures)
        with self.lock:
            ids = list(range(self.next_id, self.next_id + count))
            columns = {"id": pa.array(ids, pa.int64())}
            for name in self.keys:
                matrix = np.asarray(keys[name], np.float32).reshape(count, -1)
                columns[f"key_{name}"] = pa.FixedSizeListArray.from_arrays(pa.array(matrix.ravel()), matrix.shape[1])
            columns["exterior"] = pa.array([png_bytes(moment[0]) for moment in pictures], pa.binary())
            columns["wrist"] = pa.array([png_bytes(moment[1]) for moment in pictures], pa.binary())
            for name, kind in PAYLOAD:
                values = [payload.get(name) for payload in payloads]
                if name == "labels":
                    values = [value if isinstance(value, str) else json.dumps(value) for value in values]
                columns[name] = pa.array(values, kind)
            self.table.add(pa.table(columns, schema=self.table.schema))
            self.next_id += count
            self.count += count
            return ids

    def evict(self) -> list:
        """Remove the rows above the cap by the eviction rule. Returns the removed ids."""
        with self.lock:
            surplus = self.count - self.cap if self.cap else 0
            if surplus <= 0:
                return []
            ids = self.scan(["id"])["id"].to_numpy()
            if self.evict_rule == "fifo":
                order = np.argsort(ids, kind="stable")
            else:
                used = np.array([self.usage.get(int(i), [0, 0.0])[0] for i in ids])
                order = np.lexsort((ids, used))            # the least used first, among equally used the oldest
            gone = sorted(int(i) for i in ids[order[:surplus]])
            for start in range(0, len(gone), 500):
                self.table.delete("id IN (" + ", ".join(str(i) for i in gone[start:start + 500]) + ")")
            for i in gone:
                self.usage.pop(i, None)
                self.hits_cache.pop(i, None)
            self.count -= len(gone)
            return gone

    def flush_usage(self):
        """Write the use counters (the whole small table)."""
        with self.lock:
            ids = sorted(self.usage)
            data = pa.table({"id": pa.array(ids, pa.int64()),
                             "times_retrieved": pa.array([self.usage[i][0] for i in ids], pa.int64()),
                             "last_retrieval": pa.array([self.usage[i][1] for i in ids], pa.float64())}, schema=USAGE_SCHEMA)
            self.db.create_table("usage", data=data, schema=USAGE_SCHEMA, mode="overwrite")

    def end_update(self) -> int:
        """After an episode's insert and eviction: count the update, write counters and info, compact every 10 updates."""
        with self.lock:
            self.info["updates"] = int(self.info.get("updates", 0)) + 1
            self.flush_usage()
            if self.info["updates"] % 10 == 0:
                self.table.optimize()
            self._write_info(self.info)
            self.hits_cache.clear()
            return self.info["updates"]

    # ------------------------------------------------------------------ import, inspection
    def import_exemplar_store(self, file, instruction: str, session: str = "demo", batch: int = 200) -> int:
        """Add every entry of an exemplar store file (.npz) as a demonstration; its colour keys are recomputed with the
        task's fingerprint, which gives the file's own. Returns the number added."""
        source = ExemplarStore(self.encoder.fingerprint, self.task, str(file))
        plan = " | ".join(self.task.skills)
        added = 0
        for start in range(0, len(source), batch):
            entries = range(start, min(start + batch, len(source)))
            pictures = [[source.images[0][entry], source.images[1][entry]] for entry in entries]
            payloads = [self._demonstration_payload(source.meta[entry], instruction, session, plan) for entry in entries]
            added += len(self.insert(pictures, payloads))
        return added

    def _demonstration_payload(self, meta: dict, instruction: str, session: str, plan: str) -> dict:
        """The payload of an exemplar store entry: a demonstration, complete, labelled by its phase."""
        labels = {name: value for name, value in meta.items() if name not in ("episode", "frame", "subgoal", "phase")}
        return {"session": session, "task": self.task.name, "instruction": normalise_instruction(instruction), "plan": plan,
                "skill": self.task.skills[int(meta["subgoal"])], "subgoal": int(meta["subgoal"]),
                "phase": int(meta.get("phase", -1)), "labels": labels, "time": time.time(),
                "episode": f"demo/{int(meta['episode']):02d}", "step": int(meta["frame"]), "outcome": "complete",
                "skills_done": len(self.task.skills), "provenance": "demo", "labeller": "phase", "update_no": 0}

    def column(self, name: str) -> np.ndarray:
        """One column of every row (for analysis): a key as an (n, D) matrix."""
        data = self.scan([name])[name]
        if name.startswith("key_"):
            return np.asarray(data.combine_chunks().flatten().to_numpy(zero_copy_only=False)).reshape(len(data), -1)
        return np.asarray(data.to_pylist(), dtype=object)

    def scan(self, columns: list, where: str | None = None) -> pa.Table:
        """Columns of every row (or of the rows `where` admits), without reading the others."""
        builder = self.table.search().select(columns).limit(max(self.table.count_rows(), 1))
        if where:
            builder = builder.where(where)
        return builder.to_arrow()

    def size_bytes(self) -> dict:
        """Bytes on disk: the whole directory (all versions), and the data files the current versions reference."""
        total = sum(f.stat().st_size for f in self.path.rglob("*") if f.is_file())
        current = 0
        for name in ("moments", "usage"):
            if name in self.db.list_tables().tables:
                stats = self.db.open_table(name).stats()
                current += int(stats.get("total_bytes", 0) if isinstance(stats, dict) else getattr(stats, "total_bytes", 0))
        return {"directory": int(total), "current": int(current)}

    def close(self):
        """Write the use counters and close the query log."""
        with self.lock:
            self.flush_usage()
            if self.log_file is not None:
                self.log_file.close()
                self.log_file = None
