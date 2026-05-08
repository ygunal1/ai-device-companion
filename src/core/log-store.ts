import fs from "fs";
import path from "path";
import fetch from "node-fetch";
import dotenv from "dotenv";
import { logsDir } from "../utils/dir";
import { recognizeAudio } from "../cloud-api/server";

dotenv.config();

interface LogEntry {
  timestamp: number;
  date: string;
  type: "log" | "followup";
  transcript: string;
  participantId: string;
  deviceId: string;
}

const logFile = path.join(logsDir, "log.jsonl");
const PARTICIPANT_ID = process.env.PARTICIPANT_ID || "";
const DEVICE_ID = process.env.DEVICE_ID || "";
const TRANSCRIPT_ENDPOINT = process.env.TRANSCRIPT_ENDPOINT || "";
const TRANSCRIPT_API_KEY = process.env.TRANSCRIPT_API_KEY || "";

function appendEntry(entry: LogEntry): void {
  fs.appendFileSync(logFile, JSON.stringify(entry) + "\n", "utf-8");
}

async function sendToEndpoint(entry: LogEntry, attempt = 1): Promise<void> {
  if (!TRANSCRIPT_ENDPOINT) return;
  const maxAttempts = 4;
  const delayMs = Math.min(1000 * 2 ** (attempt - 1), 30000); // 1s, 2s, 4s, 30s cap

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
      console.error("[Log] Giving up after", maxAttempts, "attempts.");
    }
  }
}

export function saveLogEntry(params: {
  audioPath: string;
  timestamp: number;
  type?: "log" | "followup";
}): void {
  const { audioPath, timestamp, type = "log" } = params;

  recognizeAudio(audioPath)
    .then((transcript) => {
      const entry: LogEntry = {
        timestamp,
        date: new Date(timestamp).toISOString(),
        type,
        transcript: transcript || "",
        participantId: PARTICIPANT_ID,
        deviceId: DEVICE_ID,
      };
      appendEntry(entry);
      console.log(`[Log] ${type} transcript saved: "${transcript}"`);

      try {
        fs.unlinkSync(audioPath);
      } catch {
        // non-fatal
      }

      void sendToEndpoint(entry);
    })
    .catch((err) => {
      console.error("[Log] Transcription failed, saving empty transcript:", err);
      appendEntry({
        timestamp,
        date: new Date(timestamp).toISOString(),
        type,
        transcript: "",
        participantId: PARTICIPANT_ID,
        deviceId: DEVICE_ID,
      });
    });
}
