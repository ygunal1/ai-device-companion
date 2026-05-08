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
}

const logFile = path.join(logsDir, "log.jsonl");

function appendEntry(entry: LogEntry): void {
  fs.appendFileSync(logFile, JSON.stringify(entry) + "\n", "utf-8");
}

async function sendToEndpoint(entry: LogEntry): Promise<void> {
  const endpoint = process.env.TRANSCRIPT_ENDPOINT;
  if (!endpoint) return;
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(entry),
    });
    if (!res.ok) {
      console.error(`[Log] Endpoint returned ${res.status}`);
    } else {
      console.log(`[Log] Transcript sent to ${endpoint}`);
    }
  } catch (err) {
    console.error("[Log] Failed to send transcript to endpoint:", err);
  }
}

export function saveLogEntry(params: {
  audioPath: string;
  timestamp: number;
  type?: "log" | "followup";
}): void {
  const { audioPath, timestamp, type = "log" } = params;

  // Transcribe in background — do not block state transitions
  recognizeAudio(audioPath)
    .then((transcript) => {
      const entry: LogEntry = {
        timestamp,
        date: new Date(timestamp).toISOString(),
        type,
        transcript: transcript || "",
      };
      appendEntry(entry);
      console.log(`[Log] ${type} transcript saved: "${transcript}"`);

      // Delete audio file now that we have the transcript
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
      });
    });
}
