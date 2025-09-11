from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    import psutil  # optional
except Exception:
    psutil = None

_DB_DEFAULT = os.environ.get("SOLVERDASH_DB", os.path.abspath("solverdash.db"))
os.makedirs(os.path.dirname(_DB_DEFAULT), exist_ok=True)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS runs(
  run_id TEXT PRIMARY KEY,
  project TEXT,
  name TEXT,
  created_at TEXT,
  finished_at TEXT,
  status TEXT,
  notes TEXT,
  host TEXT,
  py_version TEXT,
  os TEXT,
  solver_version TEXT,
  git_commit TEXT,
  config_json TEXT
);

CREATE TABLE IF NOT EXISTS metrics(
  run_id TEXT,
  step INTEGER,
  ts TEXT,
  key TEXT,
  value REAL,
  PRIMARY KEY (run_id, step, key)
);

CREATE TABLE IF NOT EXISTS events(
  run_id TEXT,
  ts TEXT,
  level TEXT,
  message TEXT
);

CREATE TABLE IF NOT EXISTS timings(
  run_id TEXT,
  key TEXT,
  seconds REAL,
  count INTEGER,
  PRIMARY KEY(run_id, key)
);

CREATE TABLE IF NOT EXISTS artifacts(
  run_id TEXT,
  name TEXT,
  type TEXT,
  path TEXT,
  size_bytes INTEGER,
  sha256 TEXT,
  created_at TEXT,
  PRIMARY KEY (run_id, name)
);

CREATE TABLE IF NOT EXISTS tags(
  run_id TEXT,
  tag TEXT,
  PRIMARY KEY (run_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_metrics_run_step ON metrics(run_id, step);
CREATE INDEX IF NOT EXISTS idx_events_run_ts ON events(run_id, ts);
"""

def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=30.0, isolation_level=None, check_same_thread=False)
    con.execute("PRAGMA foreign_keys=ON;")
    return con

def _ensure_schema(con: sqlite3.Connection) -> None:
    for stmt in filter(None, _SCHEMA.split(";")):
        s = stmt.strip()
        if s:
            con.execute(s + ";")

@dataclasses.dataclass
class RunInfo:
    run_id: str
    project: str
    name: str
    created_at: str

class Run:
    """
    Local, threadsafe run logger writing to SQLite.
    Typical usage:
        run = start_run(project="nlp", name="try-001", config={"lr":1e-2})
        run.log_metrics(step, f=..., theta=..., stat=..., ...)
        run.log_event("started TR subproblem")
        run.log_timing("qp_solve_sec", 0.012)
        run.log_artifact("plots/trace.png", type="plot")
        run.finish(status="completed")
    """
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        db_path: str = _DB_DEFAULT,
        config: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
        solver_version: Optional[str] = None,
        git_commit: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> None:
        self.db_path = db_path
        self.con = _connect(db_path)
        _ensure_schema(self.con)
        self._lock = threading.Lock()

        now = dt.datetime.utcnow().isoformat()
        self.run_id = uuid.uuid4().hex
        self.project = project
        self.name = name or f"run-{self.run_id[:8]}"
        host = platform.node()
        py_version = platform.python_version()
        os_str = f"{platform.system()}-{platform.release()}"
        cfg_json = json.dumps(config or {}, ensure_ascii=False)
        self._insert(
            "INSERT INTO runs(run_id,project,name,created_at,finished_at,status,notes,host,py_version,os,solver_version,git_commit,config_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                self.run_id, project, self.name, now, None, "running", notes or "",
                host, py_version, os_str, solver_version or "", git_commit or "", cfg_json
            )
        )
        if tags:
            for t in tags:
                self._insert("INSERT OR IGNORE INTO tags(run_id, tag) VALUES (?,?)", (self.run_id, str(t)))

        # lightweight system logger thread (optional)
        self._syslog_stop = threading.Event()
        self._syslog_thread: Optional[threading.Thread] = None
        if psutil is not None:
            self._syslog_thread = threading.Thread(target=self._syslogger_loop, daemon=True)
            self._syslog_thread.start()

    # ------------------------ low-level helpers ------------------------ #
    def _insert(self, sql: str, params: Tuple[Any, ...]) -> None:
        with self._lock:
            self.con.execute(sql, params)

    # ------------------------ public API ------------------------------- #
    def log_metrics(self, step: int, **kv: float) -> None:
        ts = dt.datetime.utcnow().isoformat()
        rows = [(self.run_id, int(step), ts, k, float(v)) for k, v in kv.items()]
        with self._lock:
            self.con.executemany("INSERT OR REPLACE INTO metrics(run_id,step,ts,key,value) VALUES (?,?,?,?,?)", rows)

    def log_event(self, message: str, level: str = "INFO") -> None:
        ts = dt.datetime.utcnow().isoformat()
        self._insert("INSERT INTO events(run_id,ts,level,message) VALUES (?,?,?,?)",
                     (self.run_id, ts, level.upper(), message))

    @contextlib.contextmanager
    def timeit(self, key: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.log_timing(key, time.perf_counter() - t0)

    def log_timing(self, key: str, seconds: float) -> None:
        with self._lock:
            cur = self.con.execute("SELECT seconds, count FROM timings WHERE run_id=? AND key=?", (self.run_id, key))
            row = cur.fetchone()
            if row is None:
                self.con.execute("INSERT INTO timings(run_id,key,seconds,count) VALUES (?,?,?,?)",
                                 (self.run_id, key, float(seconds), 1))
            else:
                total, cnt = float(row[0]), int(row[1])
                self.con.execute("UPDATE timings SET seconds=?, count=? WHERE run_id=? AND key=?",
                                 (total + float(seconds), cnt + 1, self.run_id, key))

    def log_artifact(self, path: str, name: Optional[str] = None, type: Optional[str] = None) -> None:
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        size = os.path.getsize(path)
        sha = _sha256_file(path)
        now = dt.datetime.utcnow().isoformat()
        name = name or os.path.basename(path)
        self._insert(
            "INSERT OR REPLACE INTO artifacts(run_id,name,type,path,size_bytes,sha256,created_at) VALUES (?,?,?,?,?,?,?)",
            (self.run_id, name, type or "", path, int(size), sha, now)
        )

    def add_tags(self, *tags: str) -> None:
        with self._lock:
            self.con.executemany("INSERT OR IGNORE INTO tags(run_id, tag) VALUES (?,?)",
                                 [(self.run_id, t) for t in tags])

    def finish(self, status: str = "completed") -> None:
        self._syslog_stop.set()
        if self._syslog_thread and self._syslog_thread.is_alive():
            self._syslog_thread.join(timeout=1.0)
        now = dt.datetime.utcnow().isoformat()
        self._insert("UPDATE runs SET finished_at=?, status=? WHERE run_id=?",
                     (now, status, self.run_id))
        with self._lock:
            self.con.commit()
            self.con.close()

    # ------------------------ optional system logger ------------------- #
    def _syslogger_loop(self) -> None:
        # record basic CPU/RAM as events every ~5s
        while not self._syslog_stop.is_set():
            try:
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory().percent
                self.log_event(f"system cpu={cpu:.1f}% mem={mem:.1f}%", level="SYS")
            except Exception:
                pass
            self._syslog_stop.wait(5.0)

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def start_run(
    project: str,
    name: Optional[str] = None,
    *,
    db_path: str = _DB_DEFAULT,
    config: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
    solver_version: Optional[str] = None,
    git_commit: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
) -> Run:
    return Run(project, name, db_path, config, notes, solver_version, git_commit, tags)

@contextlib.contextmanager
def with_run(*args, **kwargs):
    run = start_run(*args, **kwargs)
    try:
        yield run
    finally:
        # only mark finished if still open
        try:
            run.finish("completed")
        except Exception:
            pass

def find_runs(db_path: str = _DB_DEFAULT, project: Optional[str] = None) -> Iterable[RunInfo]:
    con = _connect(db_path)
    _ensure_schema(con)
    cur = con.execute(
        "SELECT run_id, project, name, created_at FROM runs WHERE (? IS NULL OR project = ?) ORDER BY created_at DESC",
        (project, project))
    for rid, proj, name, created in cur.fetchall():
        yield RunInfo(rid, proj, name, created)
    con.close()
