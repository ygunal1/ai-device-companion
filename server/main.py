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
            CREATE TABLE IF NOT EXISTS logs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at   TEXT NOT NULL,
                timestamp     INTEGER,
                date          TEXT,
                participant_id TEXT,
                device_id     TEXT,
                type          TEXT,
                transcript    TEXT
            )
        """)


class LogEntry(BaseModel):
    timestamp: int
    date: str
    type: str
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
               (received_at, timestamp, date, participant_id, device_id, type, transcript)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.utcnow().isoformat(),
                entry.timestamp,
                entry.date,
                entry.participantId,
                entry.deviceId,
                entry.type,
                entry.transcript,
            ),
        )
    return {"status": "ok"}


@app.get("/export.csv")
def export_csv(_: str = Depends(require_api_key)):
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, received_at, date, participant_id, device_id, type, transcript
               FROM logs ORDER BY timestamp ASC"""
        ).fetchall()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "received_at", "date", "participant_id", "device_id", "type", "transcript"])
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
    body { font-family: sans-serif; max-width: 600px; margin: 60px auto; padding: 0 20px; color: #222; }
    h1 { font-size: 1.4rem; }
    input { padding: 8px; font-size: 1rem; width: 200px; border: 1px solid #ccc; border-radius: 4px; }
    button { padding: 8px 16px; font-size: 1rem; border: none; border-radius: 4px; cursor: pointer; margin-left: 8px; }
    #view-btn { background: #0066cc; color: white; }
    #delete-btn { background: #cc2200; color: white; display: none; }
    #result { margin-top: 24px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
    th { background: #f5f5f5; }
    .empty { color: #888; font-style: italic; }
    .confirm { background: #fff3cd; padding: 12px; border-radius: 4px; margin-top: 16px; }
  </style>
</head>
<body>
  <h1>My Study Data</h1>
  <p>Enter your participant ID to view or delete the entries recorded on your device.</p>
  <div>
    <input id="pid" type="text" placeholder="e.g. P01" />
    <button id="view-btn" onclick="loadData()">View my entries</button>
  </div>
  <div id="result"></div>

  <script>
    let currentPid = "";

    async function loadData() {
      currentPid = document.getElementById("pid").value.trim();
      if (!currentPid) return;
      const res = await fetch("/my-data/" + encodeURIComponent(currentPid));
      const data = await res.json();
      const entries = data.entries;
      const result = document.getElementById("result");

      if (entries.length === 0) {
        result.innerHTML = "<p class='empty'>No entries found for participant ID <strong>" + currentPid + "</strong>.</p>";
        document.getElementById("delete-btn").style.display = "none";
        return;
      }

      let html = "<p>Found <strong>" + entries.length + " entries</strong> for participant <strong>" + currentPid + "</strong>:</p>";
      html += "<table><tr><th>Date</th><th>Type</th><th>Transcript</th></tr>";
      for (const e of entries) {
        const d = new Date(e.date).toLocaleString();
        html += "<tr><td>" + d + "</td><td>" + e.type + "</td><td>" + (e.transcript || "<em>empty</em>") + "</td></tr>";
      }
      html += "</table>";
      html += "<div class='confirm'><button id='delete-btn' style='display:inline-block' onclick='confirmDelete()'>Request deletion of all my entries</button></div>";
      result.innerHTML = html;
    }

    async function confirmDelete() {
      if (!confirm("Delete all " + currentPid + "'s entries permanently? This cannot be undone.")) return;
      const res = await fetch("/my-data/" + encodeURIComponent(currentPid), { method: "DELETE" });
      const data = await res.json();
      document.getElementById("result").innerHTML = "<p>All entries for <strong>" + currentPid + "</strong> have been deleted.</p>";
    }
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/my-data/{participant_id}")
def get_my_data(participant_id: str):
    """Return a participant's own entries (no API key required — participants use this)."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, date, type, transcript
               FROM logs WHERE participant_id = ? ORDER BY timestamp ASC""",
            (participant_id,),
        ).fetchall()
    return {"participant_id": participant_id, "entries": [dict(r) for r in rows]}


@app.delete("/my-data/{participant_id}")
def delete_my_data(participant_id: str):
    """Delete all entries for a participant (GDPR right to erasure)."""
    with get_db() as conn:
        conn.execute("DELETE FROM logs WHERE participant_id = ?", (participant_id,))
    return {"status": "deleted", "participant_id": participant_id}


@app.get("/health")
def health():
    return {"status": "ok"}
