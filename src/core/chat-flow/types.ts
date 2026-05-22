import { StreamResponser } from "../StreamResponsor";
import { WakeWordListener } from "../../device/wakeword";

export type FlowName =
  | "sleep"
  | "camera"
  | "music"
  | "listening"
  | "wake_listening"
  | "asr"
  | "answer"
  | "image"
  | "external_answer"
  | "wake_log_listening"
  | "log_listening"
  | "log_processing"
  | "log_response"
  | "log_dynamic_followup_response"
  | "log_dynamic_followup_wait"
  | "log_dynamic_followup_listening"
  | "log_followup_wait"
  | "log_followup_listening"
  | "log_followup_response"
  | "log_followup_2_wait"
  | "log_followup_2_listening"
  | "log_confirmation"
  | "eod_prompt"
  | "eod_wait"
  | "eod_listening"
  | "eod_dynamic_followup_response"
  | "eod_dynamic_followup_wait"
  | "eod_dynamic_followup_listening"
  | "eod_followup_1"
  | "eod_followup_wait"
  | "eod_followup_listening"
  | "eod_followup_2"
  | "eod_followup_2_wait"
  | "eod_followup_2_listening"
  | "eod_confirmation";

export type FlowStateHandler = (ctx: ChatFlowContext) => void;

export interface ChatFlowContext {
  currentFlowName: FlowName;
  recordingsDir: string;
  currentRecordFilePath: string;
  asrText: string;
  streamResponser: StreamResponser;
  partialThinking: string;
  thinkingSentences: string[];
  answerId: number;
  enableCamera: boolean;
  knowledgePrompts: string[];
  wakeSessionActive: boolean;
  wakeSessionStartAt: number;
  wakeSessionLastSpeechAt: number;
  wakeSessionIdleTimeoutMs: number;
  wakeRecordMaxSec: number;
  wakeEndKeywords: string[];
  endAfterAnswer: boolean;
  pendingExternalReply: string;
  pendingExternalEmoji: string;
  pendingExternalImageUrl: string;
  currentExternalEmoji: string;
  isFromWakeListening: boolean;
  enterMusicAfterAnswer: boolean;
  musicDisplayText: string;
  wakeWordListener: WakeWordListener | null;
  pendingLogResponseText: string;
  logInitialTranscript: string;
  logLastDynamicFollowup: string;
  logLogType: "TASK-A" | "TASK-B" | "THINKING-A" | "THINKING-B" | "SOCIAL" | "REFLECTION";
  transitionTo: (flowName: FlowName) => void;
  recognizeAudio: (path: string, isFromAutoListening?: boolean) => Promise<string>;
  partialThinkingCallback: (partialThinking: string) => void;
  startWakeSession: () => void;
  endWakeSession: () => void;
  shouldContinueWakeSession: () => boolean;
  shouldEndAfterAnswer: (text: string) => boolean;
  streamExternalReply: (text: string, emoji?: string) => Promise<void>;
}
