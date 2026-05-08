import fs from "fs";
import path from "path";
import { logsDir } from "../utils/dir";

interface LogEntry {
  timestamp: number;
  date: string;
  audioPath: string;
}

const logFile = path.join(logsDir, "log.jsonl");

export function saveLogEntry(entry: { audioPath: string; timestamp: number }): void {
  try {
    const logEntry: LogEntry = {
      timestamp: entry.timestamp,
      date: new Date(entry.timestamp).toISOString(),
      audioPath: entry.audioPath,
    };
    fs.appendFileSync(logFile, JSON.stringify(logEntry) + "\n", "utf-8");
    console.log(`[Log] Saved entry: ${logEntry.date}`);
  } catch (err) {
    console.error("[Log] Failed to save log entry:", err);
  }
}
