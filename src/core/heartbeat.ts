import fetch from "node-fetch";
import dotenv from "dotenv";

dotenv.config();

const TRANSCRIPT_ENDPOINT = process.env.TRANSCRIPT_ENDPOINT || "";
const TRANSCRIPT_API_KEY = process.env.TRANSCRIPT_API_KEY || "";
const DEVICE_ID = process.env.DEVICE_ID || "";
const PARTICIPANT_ID = process.env.PARTICIPANT_ID || "";
const INTERVAL_MS = 10 * 60 * 1000; // every 10 minutes

function heartbeatEndpoint(): string {
  return TRANSCRIPT_ENDPOINT.replace(/\/logs$/, "/heartbeat");
}

function sendHeartbeat(status: string): void {
  const endpoint = heartbeatEndpoint();
  if (!endpoint || !DEVICE_ID) return;

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (TRANSCRIPT_API_KEY) headers["X-API-Key"] = TRANSCRIPT_API_KEY;

  fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify({
      deviceId: DEVICE_ID,
      participantId: PARTICIPANT_ID,
      status,
      timestamp: Date.now(),
    }),
  }).catch(() => {}); // silent — never let heartbeat failures affect the chatbot
}

export function startHeartbeat(getStatus: () => string): void {
  if (!TRANSCRIPT_ENDPOINT || !DEVICE_ID) return;
  sendHeartbeat(getStatus());
  setInterval(() => sendHeartbeat(getStatus()), INTERVAL_MS);
}
