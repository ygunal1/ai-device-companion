import os
import sqlite3
import csv
import io
from datetime import datetime
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import StreamingResponse
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


@app.get("/health")
def health():
    return {"status": "ok"}
