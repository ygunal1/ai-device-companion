import fs from "fs";
import path from "path";
import crypto from "crypto";
import fetch from "node-fetch";
import dotenv from "dotenv";
import { logsDir } from "../utils/dir";
import { recognizeAudio } from "../cloud-api/server";

dotenv.config();

interface LogEntry {
  entryId: string;
  timestamp: number;
  date: string;
  type: "log" | "followup" | "eod";
  log_type?: "TASK" | "THINKING" | "SOCIAL";
  question?: string;
  transcript: string;
  participantId: string;
  deviceId: string;
}

const logFile = path.join(logsDir, "log.jsonl");
const pendingFile = path.join(logsDir, "pending.jsonl");
const PARTICIPANT_ID = process.env.PARTICIPANT_ID || "";
const DEVICE_ID = process.env.DEVICE_ID || "";
const TRANSCRIPT_ENDPOINT = process.env.TRANSCRIPT_ENDPOINT || "";
const TRANSCRIPT_API_KEY = process.env.TRANSCRIPT_API_KEY || "";

function appendEntry(entry: LogEntry): void {
  fs.appendFileSync(logFile, JSON.stringify(entry) + "\n", "utf-8");
}

function saveToPending(entry: LogEntry): void {
  fs.appendFileSync(pendingFile, JSON.stringify(entry) + "\n", "utf-8");
  console.log("[Log] Entry saved to pending queue for retry on next startup.");
}

async function sendToEndpoint(entry: LogEntry, attempt = 1): Promise<void> {
  if (!TRANSCRIPT_ENDPOINT) return;
  const maxAttempts = 4;
  const delayMs = Math.min(1000 * 2 ** (attempt - 1), 30000);

  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (TRANSCRIPT_API_KEY) headers["X-API-Key"] = TRANSCRIPT_API_KEY;

    const res = await fetch(TRANSCRIPT_ENDPOINT, {
      method: "POST",
      headers,
      body: JSON.stringify(entry),
    });

    if (res.ok) {
      console.log(`[Log] Sent to endpoint (attempt ${attempt})`);
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (err) {
    console.error(`[Log] Endpoint send failed (attempt ${attempt}):`, err);
    if (attempt < maxAttempts) {
      setTimeout(() => sendToEndpoint(entry, attempt + 1), delayMs);
    } else {
      console.error("[Log] Max attempts reached — saving to pending queue.");
      saveToPending(entry);
    }
  }
}

// On startup, retry any entries that failed in a previous session
function retryPending(): void {
  if (!TRANSCRIPT_ENDPOINT || !fs.existsSync(pendingFile)) return;
  const lines = fs.readFileSync(pendingFile, "utf-8").split("\n").filter(Boolean);
  if (lines.length === 0) return;

  console.log(`[Log] Retrying ${lines.length} pending entries from previous session...`);
  fs.writeFileSync(pendingFile, "", "utf-8");

  for (const line of lines) {
    try {
      void sendToEndpoint(JSON.parse(line) as LogEntry);
    } catch {
      // malformed line — skip
    }
  }
}

// Wait 10s after startup for network to be ready before retrying
if (TRANSCRIPT_ENDPOINT) {
  setTimeout(retryPending, 10000);
}

export function saveLogEntry(params: {
  audioPath: string;
  timestamp: number;
  type?: "log" | "followup" | "eod";
  log_type?: "TASK" | "THINKING" | "SOCIAL";
  question?: string;
}): Promise<string> {
  const { audioPath, timestamp, type = "log", log_type, question } = params;

  const deleteAudio = () => {
    try { fs.unlinkSync(audioPath); } catch {}
  };

  return recognizeAudio(audioPath)
    .then((transcript) => {
      const entry: LogEntry = {
        entryId: crypto.randomUUID(),
        timestamp,
        date: new Date(timestamp).toISOString(),
        type,
        ...(log_type ? { log_type } : {}),
        ...(question ? { question } : {}),
        transcript: transcript || "",
        participantId: PARTICIPANT_ID,
        deviceId: DEVICE_ID,
      };
      appendEntry(entry);
      deleteAudio();
      console.log(`[Log] ${type} transcript saved: "${transcript}"`);
      void sendToEndpoint(entry);
      return transcript || "";
    })
    .catch((err) => {
      console.error("[Log] Transcription failed:", err);
      appendEntry({
        entryId: crypto.randomUUID(),
        timestamp,
        date: new Date(timestamp).toISOString(),
        type,
        ...(question ? { question } : {}),
        transcript: "",
        participantId: PARTICIPANT_ID,
        deviceId: DEVICE_ID,
      });
      deleteAudio();
      return "";
    });
}
