import os
import sqlite3
import csv
import io
from datetime import datetime
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

DB_PATH = os.environ.get("DB_PATH", "logs.db")
API_KEY = os.environ.get("API_KEY", "")

# Per-participant passwords: "P01:pass1,P02:pass2" in env var
_raw = os.environ.get("PARTICIPANT_PASSWORDS", "")
PARTICIPANT_PASSWORDS: dict[str, str] = dict(
    item.split(":", 1) for item in _raw.split(",") if ":" in item
)


def check_participant_auth(participant_id: str, password: str) -> None:
    if not PARTICIPANT_PASSWORDS:
        return  # no passwords configured — open access
    expected = PARTICIPANT_PASSWORDS.get(participant_id)
    if expected is None or password != expected:
        raise HTTPException(status_code=403, detail="Invalid participant ID or password")

app = FastAPI(title="Whisplay Log Collector")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str = Security(api_key_header)):
    if not API_KEY:
        raise RuntimeError("API_KEY env var not set on server")
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS heartbeats (
                device_id     TEXT PRIMARY KEY,
                participant_id TEXT,
                last_seen     TEXT NOT NULL,
                status        TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at   TEXT NOT NULL,
                timestamp     INTEGER,
                date          TEXT,
                participant_id TEXT,
                device_id     TEXT,
                type          TEXT,
                question      TEXT,
                transcript    TEXT,
                deleted_at    TEXT
            )
        """)
        # Migrations: add columns to existing databases that don't have them
        cols = [r[1] for r in conn.execute("PRAGMA table_info(logs)").fetchall()]
        if "deleted_at" not in cols:
            conn.execute("ALTER TABLE logs ADD COLUMN deleted_at TEXT")
        if "question" not in cols:
            conn.execute("ALTER TABLE logs ADD COLUMN question TEXT")


class Heartbeat(BaseModel):
    deviceId: str
    participantId: str = ""
    status: str = "unknown"
    timestamp: int = 0


class LogEntry(BaseModel):
    timestamp: int
    date: str
    type: str
    question: str = ""
    transcript: str
    participantId: str = ""
    deviceId: str = ""


@app.on_event("startup")
def startup():
    init_db()


@app.post("/logs", status_code=201)
def receive_log(entry: LogEntry, _: str = Depends(require_api_key)):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO logs
               (received_at, timestamp, date, participant_id, device_id, type, question, transcript)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.utcnow().isoformat(),
                entry.timestamp,
                entry.date,
                entry.participantId,
                entry.deviceId,
                entry.type,
                entry.question or None,
                entry.transcript,
            ),
        )
    return {"status": "ok"}


@app.post("/heartbeat", status_code=200)
def receive_heartbeat(hb: Heartbeat, _: str = Depends(require_api_key)):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO heartbeats (device_id, participant_id, last_seen, status)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(device_id) DO UPDATE SET
                 participant_id = excluded.participant_id,
                 last_seen      = excluded.last_seen,
                 status         = excluded.status""",
            (hb.deviceId, hb.participantId, datetime.utcnow().isoformat(), hb.status),
        )
    return {"status": "ok"}


@app.get("/devices")
def list_devices(_: str = Depends(require_api_key)):
    """Show all devices and when they last checked in."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT device_id, participant_id, last_seen, status FROM heartbeats ORDER BY last_seen DESC"
        ).fetchall()
    return {"devices": [dict(r) for r in rows]}


@app.get("/export.csv")
def export_csv(from_date: str = "", to_date: str = "", _: str = Depends(require_api_key)):
    filters = ["1=1"]
    params: list = []
    if from_date:
        filters.append("date >= ?")
        params.append(from_date)
    if to_date:
        filters.append("date <= ?")
        params.append(to_date + "T23:59:59")
    where = " AND ".join(filters)
    with get_db() as conn:
        rows = conn.execute(
            f"""SELECT id, received_at, date, participant_id, device_id, type, question, transcript, deleted_at
               FROM logs WHERE {where} ORDER BY timestamp ASC""",
            params,
        ).fetchall()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "received_at", "date", "participant_id", "device_id", "type", "question", "transcript", "deleted_at"])
    for row in rows:
        writer.writerow(list(row))

    buf.seek(0)
    filename = f"whisplay_logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── Participant self-service ──────────────────────────────────────────────────

@app.get("/my-data", response_class=HTMLResponse)
def my_data_page():
    """Simple HTML page participants can use to view or delete their entries."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>My Study Data</title>
  <style>
    body { font-family: sans-serif; max-width: 680px; margin: 60px auto; padding: 0 20px; color: #222; }
    h1 { font-size: 1.4rem; }
    input { padding: 8px; font-size: 1rem; width: 200px; border: 1px solid #ccc; border-radius: 4px; }
    button { padding: 8px 16px; font-size: 1rem; border: none; border-radius: 4px; cursor: pointer; margin-left: 8px; }
    #view-btn { background: #0066cc; color: white; }
    #result { margin-top: 24px; }
    .fields { display: flex; flex-direction: column; gap: 10px; max-width: 340px; }
    .fields label { font-size: 0.9rem; color: #555; margin-bottom: 2px; display: block; }
    .fields input { width: 100%; box-sizing: border-box; margin: 0; }
    .fields button { margin: 4px 0 0 0; }
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; vertical-align: top; }
    th { background: #f5f5f5; }
    .empty { color: #888; font-style: italic; }
    .del-entry { background: #cc2200; color: white; font-size: 0.8rem; padding: 4px 10px; margin-left: 0; }
    .del-all { background: #7a0000; color: white; margin-top: 16px; }
    .error { color: #cc2200; margin-top: 12px; }
  </style>
</head>
<body>
  <h1>My Study Data</h1>
  <p>Enter your participant ID and password to view or remove entries recorded on your device.</p>
  <div class="fields">
    <div>
      <label for="pid">Participant ID</label>
      <input id="pid" type="text" placeholder="e.g. P01" />
    </div>
    <div>
      <label for="pw">Password</label>
      <input id="pw" type="password" placeholder="your password" onkeydown="if(event.key==='Enter')loadData()" />
    </div>
    <div>
      <button id="view-btn" onclick="loadData()">View my entries</button>
    </div>
  </div>
  <div id="result"></div>

  <script>
    let currentPid = "";
    let currentPw = "";

    function authParams() {
      return "?password=" + encodeURIComponent(currentPw);
    }

    async function loadData() {
      currentPid = document.getElementById("pid").value.trim();
      currentPw  = document.getElementById("pw").value;
      if (!currentPid) return;
      const res = await fetch("/my-data/" + encodeURIComponent(currentPid) + authParams());
      if (res.status === 403) {
        document.getElementById("result").innerHTML = "<p class='error'>Incorrect participant ID or password.</p>";
        return;
      }
      const data = await res.json();
      render(data.entries);
    }

    function render(entries) {
      const result = document.getElementById("result");
      if (entries.length === 0) {
        result.innerHTML = "<p class='empty'>No entries found for participant ID <strong>" + currentPid + "</strong>.</p>";
        return;
      }

      let html = "<p>Found <strong>" + entries.length + " entries</strong> for participant <strong>" + currentPid + "</strong>.</p>";
      html += "<table><tr><th>Date</th><th>Type</th><th>Transcript</th><th></th></tr>";
      for (const e of entries) {
        const d = new Date(e.date).toLocaleString();
        html += "<tr id='row-" + e.id + "'>";
        html += "<td>" + d + "</td>";
        html += "<td>" + e.type + "</td>";
        html += "<td>" + (e.transcript || "<em>empty</em>") + "</td>";
        html += "<td><button class='del-entry' onclick='deleteEntry(" + e.id + ")'>Delete</button></td>";
        html += "</tr>";
      }
      html += "</table>";
      html += "<br><button class='del-all' onclick='deleteAll()'>Delete all my entries</button>";
      result.innerHTML = html;
    }

    async function deleteEntry(id) {
      if (!confirm("Remove this entry from your record?")) return;
      const res = await fetch("/my-data/" + encodeURIComponent(currentPid) + "/" + id + authParams(), { method: "DELETE" });
      if (res.ok) {
        const row = document.getElementById("row-" + id);
        if (row) row.remove();
        const rows = document.querySelectorAll("table tr[id^='row-']");
        if (rows.length === 0) {
          document.getElementById("result").innerHTML = "<p class='empty'>All entries removed.</p>";
        }
      } else {
        alert("Could not delete entry. Please try again.");
      }
    }

    async function deleteAll() {
      if (!confirm("Remove all entries for " + currentPid + "? This cannot be undone.")) return;
      const res = await fetch("/my-data/" + encodeURIComponent(currentPid) + authParams(), { method: "DELETE" });
      if (res.ok) {
        document.getElementById("result").innerHTML = "<p>All entries for <strong>" + currentPid + "</strong> have been removed from your view.</p>";
      } else {
        alert("Could not delete entries. Please try again.");
      }
    }
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/my-data/{participant_id}")
def get_my_data(participant_id: str, password: str = ""):
    """Return a participant's own non-deleted entries."""
    check_participant_auth(participant_id, password)
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, date, type, transcript
               FROM logs
               WHERE participant_id = ? AND deleted_at IS NULL
               ORDER BY timestamp ASC""",
            (participant_id,),
        ).fetchall()
    return {"participant_id": participant_id, "entries": [dict(r) for r in rows]}


@app.delete("/my-data/{participant_id}/{entry_id}")
def delete_my_entry(participant_id: str, entry_id: int, password: str = ""):
    """Soft-delete a single entry. Only works if it belongs to the participant."""
    check_participant_auth(participant_id, password)
    with get_db() as conn:
        result = conn.execute(
            """UPDATE logs SET deleted_at = ?
               WHERE id = ? AND participant_id = ? AND deleted_at IS NULL""",
            (datetime.utcnow().isoformat(), entry_id, participant_id),
        )
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Entry not found")
    return {"status": "deleted", "id": entry_id}


@app.delete("/my-data/{participant_id}")
def delete_my_data(participant_id: str, password: str = ""):
    """Soft-delete all entries for a participant (GDPR right to erasure)."""
    check_participant_auth(participant_id, password)
    with get_db() as conn:
        conn.execute(
            "UPDATE logs SET deleted_at = ? WHERE participant_id = ? AND deleted_at IS NULL",
            (datetime.utcnow().isoformat(), participant_id),
        )
    return {"status": "deleted", "participant_id": participant_id}


@app.get("/health")
def health():
    return {"status": "ok"}
