import moment from "moment";
import fs from "fs";
import { compact, noop } from "lodash";
import {
  onButtonPressed,
  onButtonReleased,
  onButtonDoubleClick,
  display,
  getCurrentStatus,
  onCameraCapture,
  onTextInput,
  isButtonDown,
  setFace,
} from "../../device/display";
import {
  recordAudio,
  recordAudioManually,
  recordFileFormat,
  getDynamicVoiceDetectLevel,
} from "../../device/audio";
import { chatWithLLMStream } from "../../cloud-api/server";
import { isImMode } from "../../cloud-api/llm";
import { getSystemPromptWithKnowledge } from "../Knowledge";
import { enableRAG } from "../../cloud-api/knowledge";
import { cameraDir } from "../../utils/dir";
import {
  clearPendingCapturedImgForChat,
  getLatestGenImg,
  getLatestDisplayImg,
  setLatestCapturedImg,
  setPendingCapturedImgForChat,
} from "../../utils/image";
import { sendWhisplayIMMessage } from "../../cloud-api/whisplay-im/whisplay-im";
import { ChatFlowContext, FlowName, FlowStateHandler } from "./types";
import {
  enterCameraMode,
  handleCameraModePress,
  handleCameraModeRelease,
  onCameraModeExit,
  resetCameraModeControl,
} from "./camera-mode";
import { DEFAULT_EMOJI } from "../../utils";
import { isMusicPlaying, getCurrentTrackTitle, stopMusicPlayback, startPendingMusicPlayback, onMusicTrackChange, onMusicPlaybackEnd } from "../../device/music-player";
import { autoSaveExchange } from "../../config/mempalace";
import { saveLogEntry } from "../log-store";
import { openai, openaiLLMModel } from "../../cloud-api/openai/openai";

const LONG_PRESS_MS = parseInt(process.env.LONG_PRESS_MS || "1500");
const FOLLOWUP_WAIT_TIMEOUT_MS = parseInt(process.env.FOLLOWUP_WAIT_TIMEOUT_MS || "60000");
const SLEEP_DISPLAY_TEXT = "Long press the button to log an entry.";

const audioHasContent = (filePath: string): boolean => {
  try {
    return fs.existsSync(filePath) && fs.statSync(filePath).size > 500;
  } catch {
    return false;
  }
};

const handleEmptyAudio = (ctx: ChatFlowContext, returnState: FlowName): void => {
  ctx.streamResponser.stop();
  display({ status: "answering...", emoji: "😕", RGB: "#ff6600", text: "I didn't catch that. Please try again." });
  void ctx.streamExternalReply("Sorry, I didn't catch that. Please try again.");
  ctx.streamResponser.getPlayEndPromise().then(() => {
    ctx.transitionTo(returnState);
  });
};

type LogType = "TASK" | "THINKING" | "SOCIAL" | "REFLECTION";

const FOLLOWUP_1: Record<LogType, string> = {
  TASK: "How useful would it be for me to take care of this for you and why?",
  THINKING: "How useful would it be for me to help with something like this and why?",
  SOCIAL: "How useful would it be for me to help with something like this and why?",
  REFLECTION: "How useful would it be for me to help with something like this and why?",
};
const FOLLOWUP_1_WITH_TRANSITION: Record<LogType, string> = {
  TASK: "Got it. How useful would it be for me to take care of this for you and why?",
  THINKING: "Got it. How useful would it be for me to help with something like this and why?",
  SOCIAL: "Got it. How useful would it be for me to help with something like this and why?",
  REFLECTION: "Got it. How useful would it be for me to help with something like this and why?",
};
const FOLLOWUP_2: Record<LogType, string> = {
  TASK: "What tools do you normally use for this?",
  THINKING: "What would you normally do about something like this?",
  SOCIAL: "What would you normally do about something like this?",
  REFLECTION: "What would you normally do about something like this?",
};
const LOG_CONFIRMATION = "Got it, I've noted that down.";

const DEVICE_PERSONALITY_PROMPT = `You are a warm, supportive voice assistant helping a knowledge worker
capture their thoughts during the workday.

Your personality:
- Warm and polite, but not overly effusive or emotional
- Empathetic when participants express frustration or difficulty,
  but do not dwell on it — acknowledge briefly and move on
- Professional and calm — you are a work tool, not a therapist or friend
- Never use filler phrases like "Great!", "Absolutely!", or "Of course!"
  — these feel hollow and robotic
- Keep responses concise — you are speaking aloud to someone mid-task

When a participant expresses a negative emotion or frustration:
- Acknowledge it briefly with one short phrase before asking your question
- Example: "That sounds frustrating." or "I can see why that would be
  annoying." — then follow immediately with the question
- Do not ask about the emotion itself unless it is directly relevant to
  understanding the log

When confirming a log:
- Use natural, warm but brief confirmations: "Got it." / "Noted." /
  "I've got that." — not clinical ("Recorded.") or overly enthusiastic
  ("Great, I've noted that down for you!")

Tone calibration:
- More warm than neutral, but more professional than personal
- Think: a calm, competent colleague who listens well — not a chatbot,
  not a therapist, not a smart speaker`;

const DYNAMIC_FOLLOWUP_SYSTEM_PROMPT = `You are a warm, professional voice assistant helping a researcher collect
structured diary logs from knowledge workers. A participant has just spoken
a short voice log describing something they wished an AI agent could help
them with during their workday.

Your job has two parts: classify the log, then decide whether to ask a
follow-up question or return null.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — CLASSIFY THE LOG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Classify into exactly one of the following types.

TASK — a concrete, delegatable request with a clear output
Examples: reminders, summaries, drafts, notifications, searches,
scheduling, code review, finding information, taking notes

THINKING — reasoning, deciding, or problem-solving out loud
Examples: comparing options ("X or Y"), being stuck, working through
a decision, exploring tradeoffs, preparing for a conversation
IMPORTANT: Any log containing "X or Y", "better", "should I",
"which one", "what’s the difference", or similar comparative or
decision language is ALWAYS classified as THINKING — never TASK.

SOCIAL — navigating a relationship or interpersonal situation
Examples: giving feedback, handling conflict, a difficult conversation,
collaboration friction, reviewing a colleague\’s work

REFLECTION — retrospective, processing something that already happened
Examples: a meeting that went badly, a week of low productivity,
capturing lessons from an incident, wishing they had done something
differently

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — DYNAMIC FOLLOW-UP DECISION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIRST — check for incoherence or truncation:
If the log starts mid-sentence, ends abruptly, is a single word or
fragment, contains only filler ("You", "Thanks", "Okay"), or cannot
be interpreted as a complete thought → return:
"sorry, I didn\’t catch that — could you say that again?"
This does not count as a dynamic follow-up.

SECOND — check for empty or nonsense responses:
If previous_response is empty, clearly nonsense, or unrelated to the
question asked → treat previous_response as if it was never given.
Re-evaluate using only the original log and re-ask the most relevant
follow-up question (may re-ask the same question).

THEN — apply the two-rule decision:

RULE 1 — Return null only if ALL of the following are true:
  - There is a clear call to action
  - No critical context is missing that the agent could not infer
  - Critical context by log type:
      TASK: people or recipients involved, specific deadlines or
        timing, external platforms or tools, meeting context,
        level of detail for informational or description requests
      THINKING: what the decision is actually about (not just how
        the participant wants help), what the options are, what
        is making it difficult
      SOCIAL: who is involved, what the situation is, what outcome
        is needed
      REFLECTION: what happened, what the participant is processing,
        what kind of support they want
  - THINKING, SOCIAL, and REFLECTION logs should almost never
    return null — there is almost always missing context

RULE 2 — Type-specific decision tree. Follow the branch for the log type.
Do NOT skip to the fallback unless every branch for that type has been
exhausted. The fallback must NEVER be reached for THINKING, SOCIAL, or
REFLECTION — those types always have a relevant question to ask.

THINKING logs:
  a. No decision mentioned at all →
       "what decision are you working through?"
  b. Two options named but no driving context (e.g. "X or Y", "A or B",
     "fork or clone", "which is better") →
       "what\’s making the [option A] vs [option B] decision difficult?"
     CRITICAL: ANY log with "X or Y", "A or B", "which", "better",
     "should I", or two named options ALWAYS hits branch (b).
     Never fall through to the fallback.
  c. Decision described but consequences or stakes unclear →
       "what\’s at stake with this one?"
  d. Stuck or uncertain but no reason given →
       "what\’s making this hard to figure out?"

SOCIAL logs:
  a. No person named → "who\’s involved in this situation?"
  b. Person named but situation not described →
       "what\’s been happening with [person]?"
  c. Person + situation named but participant\’s goal is unclear →
       "what outcome are you hoping for from this?"
     CRITICAL: If a person and situation are BOTH present in the log,
     you must use branch (c). Never fall through to the fallback.
  d. Conflict with no prior attempts described →
       "what have you already tried with this?"

REFLECTION logs:
  a. No event described → "what happened?"
  b. Event described (even briefly) →
       "what do you think\’s been driving that?"
     CRITICAL: If the log describes ANY event or situation that already
     happened, you must use branch (b). Never fall through to the fallback.
  c. Event + possible cause described, but processing is unclear →
       "what part of this are you most trying to work through?"

TASK logs:
  a. Missing recipient or target → ask who or what
  b. Missing timing or deadline → ask when
  c. Informational request without detail level →
       "quick summary or a detailed overview?"
  d. Missing platform or tool → ask where or how
  e. Context fully inferable → return null

FALLBACK (only if no type-specific branch above applies):
  "how would you want me to help with this?"
  IMPORTANT: This must never be returned for THINKING, SOCIAL, or
  REFLECTION. If you are about to return the fallback for one of those
  types, stop and re-read the type-specific branches above.

Additional rules:
- Maximum 2 dynamic follow-ups per log entry
- After a follow-up is answered with a valid response,
  re-evaluate using the original log plus all previous responses
  together. Only ask a second follow-up if a gap existed in the
  original log and remains unresolved. Do not chase new gaps
  introduced by the response itself.
- When in doubt → return null for TASK; ask for THINKING,
  SOCIAL, REFLECTION

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — STATIC FOLLOW-UPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After 0–2 dynamic follow-ups, always populate the static follow-ups:

If TASK:
  static_followup_1: "How useful would it be for me to take care
    of this for you, and why?"
  static_followup_2: "What tools do you normally use for this?"

If THINKING, SOCIAL, or REFLECTION:
  static_followup_1: "How useful would it be for me to help with
    something like this, and why?"
  static_followup_2: "What would you normally do about something
    like this?"

confirmation (all types): "Got it, I\’ve noted that down."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Example 1 — THINKING (branch b):
  Log: "Should I do X or Y?"
  → log_type: THINKING
  → dynamic_followup: "what\'s making the X vs Y decision difficult?"

Example 2 — THINKING (branch b):
  Log: "Help me decide whether to use approach A or approach B."
  → log_type: THINKING
  → dynamic_followup: "what\'s making the approach A vs approach B decision difficult?"

Example 3 — TASK (branch c, informational):
  Log: "What is the order of operations for deploying code to AWS?"
  → log_type: TASK
  → dynamic_followup: "quick summary or a detailed overview?"

Example 4 — TASK (branch c, informational):
  Log: "Describe to me what function ABC does."
  → log_type: TASK
  → dynamic_followup: "quick summary or a detailed overview?"

Example 5 — REFLECTION (branch b):
  Log: "I have been less productive this week than usual and I\'m not sure why."
  → log_type: REFLECTION
  → dynamic_followup: "what do you think\'s been driving that?"

Example 6 — TASK (null, context fully inferable):
  Log: "Remind me to ping my teammate at 2 PM."
  → log_type: TASK
  → dynamic_followup: null

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELF-CHECK (required before writing JSON)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before writing your JSON output, verify your chosen dynamic_followup:

1. If log_type is THINKING and dynamic_followup is "how would you
   want me to help with this?" → ERROR. This fallback is forbidden
   for THINKING. Replace it with the correct branch question.
   If you are unsure which branch applies, default to:
   "what decision are you working through?"

2. If log_type is SOCIAL and dynamic_followup is "how would you
   want me to help with this?" → ERROR. This fallback is forbidden
   for SOCIAL. Replace it with the correct branch question.
   If you are unsure which branch applies, default to:
   "who\'s involved in this situation?"

3. If log_type is REFLECTION and dynamic_followup is "how would you
   want me to help with this?" → ERROR. This fallback is forbidden
   for REFLECTION. Replace it with the correct branch question.
   If you are unsure which branch applies, default to:
   "what happened?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARD LIMITS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- No yes/no questions
- Maximum 15 words per dynamic follow-up question
- Never describe what you are doing — speak the question directly
- Never use filler: "Great!", "Sure!", "Absolutely!", "Of course!"
- Return only the question string or null for dynamic_followup
- Always return valid JSON — no explanation, no markdown, no prose

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return this JSON object and nothing else. No markdown, no backticks,
no explanation before or after.

{
  "log_type": "TASK" | "THINKING" | "SOCIAL" | "REFLECTION",
  "current_turn": "dynamic_followup" | "static_followup_1" | "static_followup_2" | "confirmation",
  "dynamic_followup": "<question>" | null,
  "static_followup_1": "<question>",
  "static_followup_2": "<question>",
  "confirmation": "Got it, I\’ve noted that down."
}

current_turn must reflect which question should be delivered now:
- If dynamic_followup is not null → current_turn is "dynamic_followup"
- If dynamic_followup is null → current_turn is "static_followup_1"`

async function generateDynamicFollowup(
  transcript: string,
  previousFollowup: string,
  previousResponse: string,
  isEod: boolean = false
): Promise<{ question: string | null; logType: LogType }> {
  const fallback = { question: null, logType: "TASK" as LogType };
  if (!openai) return fallback;

  const userContent = `Current log: "${transcript}"
Previous follow-up asked (if any): "${previousFollowup}"
Previous follow-up response (if any): "${previousResponse}"${
    isEod
      ? `\nContext: This is an end-of-day reflection. The participant is describing something retrospectively — a moment earlier in the day where they wished they had used the device.

For REFLECTION logs: ask what specifically happened or what made it difficult in the moment — not why they feel that way generally.
For TASK logs: ask when the intention arose or what triggered it.

Good EOD follow-up examples:
- "what made it hard to ask for help with that at the time?"
- "when did you first realize you needed help with it?"
- "what were you trying to figure out when you got stuck?"

Bad EOD follow-up examples (too therapeutic, too vague):
- "what do you think's been driving that?"
- "how did that make you feel?"
- "what's making it difficult?"`
      : ""
  }`;

  try {
    const completion = await openai.chat.completions.create({
      model: openaiLLMModel,
      messages: [
        { role: "system", content: DYNAMIC_FOLLOWUP_SYSTEM_PROMPT },
        { role: "user", content: userContent },
      ],
      max_tokens: 300,
      temperature: 0.3,
    });

    const raw = completion.choices[0]?.message?.content?.trim() || "{}";

    // Strip markdown code fences if model wraps response anyway
    const cleaned = raw.replace(/^```json\s*/i, "").replace(/```$/i, "").trim();

    const parsed = JSON.parse(cleaned) as {
      dynamic_followup?: string | null;
      log_type?: string;
    };

    const validLogTypes: LogType[] = ["TASK", "THINKING", "SOCIAL", "REFLECTION"];
    const logType: LogType = validLogTypes.includes(parsed.log_type as LogType)
      ? (parsed.log_type as LogType)
      : "TASK";

    let question = parsed.dynamic_followup ?? null;
    if (question) {
      // Ensure only one question — take everything up to and including the first "?"
      const firstQ = question.indexOf("?");
      if (firstQ !== -1) question = question.slice(0, firstQ + 1).trim();
    }

    // Guard: if the LLM returned the fallback for a type where it's forbidden,
    // replace with the appropriate default so we always get useful context.
    const FALLBACK_STRING = "how would you want me to help with this?";
    if (question?.toLowerCase().includes(FALLBACK_STRING) && logType !== "TASK") {
      console.warn(`[DynamicFollowup] Fallback returned for ${logType} type — overriding`);
      const defaults: Record<LogType, string> = {
        TASK: question,
        THINKING: "what decision are you working through?",
        SOCIAL: "who's involved in this situation?",
        REFLECTION: "what happened?",
      };
      question = defaults[logType];
    }

    return { question: question || null, logType };
  } catch (err) {
    console.error("[DynamicFollowup] LLM call failed:", err);
    return fallback;
  }
}

const EOD_QUESTION = "Thinking about your day, is there anything you wish you could have used me for that you haven't logged?";
const EOD_FOLLOWUP_1 = "If this comes up again how useful would it be to have me help and why?";
const EOD_FOLLOWUP_2 = "What was happening when you first realized you needed help with it?";
const EOD_CONFIRMATION = "Got it, I've noted that down, and have a good evening.";

export const flowStates: Record<FlowName, FlowStateHandler> = {
  sleep: (ctx: ChatFlowContext) => {
    let longPressTimer: NodeJS.Timeout | null = null;

    setFace("idle");
    onCameraModeExit(null);
    onButtonDoubleClick(null);

    onButtonPressed(() => {
      resetCameraModeControl();
      stopMusicPlayback();
      ctx.wakeWordListener?.stop();
      setFace("listening");
      display({ status: "listening", text: "Listening...", RGB: "#00aa44" });
      longPressTimer = setTimeout(() => {
        longPressTimer = null;
        ctx.transitionTo("log_listening");
      }, LONG_PRESS_MS);
    });

    onButtonReleased(() => {
      if (longPressTimer) {
        clearTimeout(longPressTimer);
        longPressTimer = null;
      }
      if (ctx.currentFlowName === "sleep") {
        setFace("idle");
        display({ status: "idle", emoji: "", RGB: "#000055", text: SLEEP_DISPLAY_TEXT });
      }
    });

    onTextInput((text: string) => {
      if (ctx.currentFlowName !== "sleep") return;
      ctx.answerId += 1;
      ctx.asrText = text;
      display({ status: "recognizing", text, text_input_enabled: false });
      ctx.transitionTo("answer");
    });

    display({
      status: "idle",
      emoji: "",
      RGB: "#000055",
      rag_icon_visible: false,
      text: SLEEP_DISPLAY_TEXT,
    });
  },
  camera: (ctx: ChatFlowContext) => {
    onButtonDoubleClick(null);
    onButtonPressed(() => {
      handleCameraModePress();
    });
    onButtonReleased(() => {
      handleCameraModeRelease();
    });
    onCameraCapture(() => {
      const captureImagePath = getCurrentStatus().capture_image_path;
      if (!captureImagePath) {
        return;
      }
      setLatestCapturedImg(captureImagePath);
      setPendingCapturedImgForChat(captureImagePath);
      display({ image_icon_visible: true });
    });
    onCameraModeExit(() => {
      if (ctx.currentFlowName === "camera") {
        ctx.transitionTo("sleep");
      }
    });
    display({
      status: "camera",
      emoji: "📷",
      RGB: "#00ff88",
    });
  },
  music: (ctx: ChatFlowContext) => {
    // Start deferred music playback when entering music state
    startPendingMusicPlayback();

    // Update display when track changes during continuous playback
    onMusicTrackChange((title) => {
      if (ctx.currentFlowName === "music") {
        display({ text: `Now playing: ${title}` });
      }
    });

    // Return to sleep when non-continuous playback finishes
    onMusicPlaybackEnd(() => {
      if (ctx.currentFlowName === "music") {
        onMusicTrackChange(null);
        onMusicPlaybackEnd(null);
        ctx.transitionTo("sleep");
      }
    });

    onButtonDoubleClick(null);
    onButtonPressed(() => {
      // Stop music immediately when button is pressed
      onMusicTrackChange(null);
      onMusicPlaybackEnd(null);
      stopMusicPlayback();
      ctx.transitionTo("listening");
    });
    onButtonReleased(noop);

    const trackTitle = getCurrentTrackTitle();
    display({
      status: "music",
      emoji: "🎹",
      RGB: "#0066aa",
      text:
        ctx.musicDisplayText ||
        (isMusicPlaying() && trackTitle
          ? `Now playing: ${trackTitle}`
          : "Music mode. Press the button to talk."),
      rag_icon_visible: false,
    });
  },
  listening: (ctx: ChatFlowContext) => {
    ctx.enterMusicAfterAnswer = false;
    ctx.musicDisplayText = "";
    ctx.isFromWakeListening = false;
    ctx.answerId += 1;
    ctx.wakeSessionActive = false;
    ctx.endAfterAnswer = false;
    onButtonDoubleClick(null);
    ctx.currentRecordFilePath = `${ctx.recordingsDir
      }/user-${Date.now()}.${recordFileFormat}`;
    onButtonPressed(noop);
    const listeningStartedAt = Date.now();
    // If button was already released before we entered this state, go back to sleep
    if (!isButtonDown()) {
      console.log("[listening] Button already released, returning to sleep");
      ctx.transitionTo("sleep");
      return;
    }
    const { result, stop } = recordAudioManually(ctx.currentRecordFilePath);
    const handleRelease = () => {
      if (Date.now() - listeningStartedAt < 500) {
        // Too short to be meaningful — stop recording and return to sleep
        console.log("[listening] Button released too quickly, returning to sleep");
        stop();
        ctx.transitionTo("sleep");
        return;
      }
      stop();
      display({
        RGB: "#ff6800",
      });
    };
    onButtonReleased(handleRelease);
    result
      .then(() => {
        ctx.transitionTo("asr");
      })
      .catch((err) => {
        console.error("Error during recording:", err);
        ctx.transitionTo("sleep");
      });
    display({
      status: "listening",
      emoji: DEFAULT_EMOJI,
      RGB: "#00ff00",
      text: "Listening...",
      rag_icon_visible: false,
    });
  },
  wake_listening: (ctx: ChatFlowContext) => {
    setFace("listening");
    ctx.enterMusicAfterAnswer = false;
    ctx.musicDisplayText = "";
    ctx.isFromWakeListening = true;
    ctx.answerId += 1;
    ctx.currentRecordFilePath = `${ctx.recordingsDir
      }/user-${Date.now()}.${recordFileFormat}`;
    onButtonPressed(() => {
      ctx.transitionTo("listening");
    });
    onButtonReleased(noop);
    display({
      status: "detecting",
      emoji: DEFAULT_EMOJI,
      RGB: "#00ff00",
      text: "Detecting voice level...",
      rag_icon_visible: false,
    });
    getDynamicVoiceDetectLevel().then((level) => {
      display({
        status: "listening",
        emoji: DEFAULT_EMOJI,
        RGB: "#00ff00",
        text: `(Detect level: ${level}%) Listening...`,
        rag_icon_visible: false,
      });
      recordAudio(ctx.currentRecordFilePath, ctx.wakeRecordMaxSec, level)
        .then(() => {
          ctx.transitionTo("asr");
        })
        .catch((err) => {
          console.error("Error during auto recording:", err);
          ctx.endWakeSession();
          ctx.transitionTo("sleep");
        });
    });
  },
  asr: (ctx: ChatFlowContext) => {
    display({
      status: "recognizing",
    });
    onButtonDoubleClick(null);
    Promise.race([
      ctx.recognizeAudio(ctx.currentRecordFilePath, ctx.isFromWakeListening),
      new Promise<string>((resolve) => {
        onButtonPressed(() => {
          resolve("[UserPress]");
        });
        onButtonReleased(noop);
      }),
    ]).then((result) => {
      if (ctx.currentFlowName !== "asr") return;
      if (result === "[UserPress]") {
        ctx.transitionTo("listening");
        return;
      }
      if (result) {
        console.log("Audio recognized result:", result);
        ctx.asrText = result;
        ctx.endAfterAnswer = ctx.shouldEndAfterAnswer(result);
        if (ctx.wakeSessionActive) {
          ctx.wakeSessionLastSpeechAt = Date.now();
        }
        display({ status: "recognizing", text: result });
        ctx.transitionTo("answer");
        return;
      }
      if (ctx.wakeSessionActive) {
        if (ctx.shouldContinueWakeSession()) {
          ctx.transitionTo("wake_listening");
        } else {
          ctx.endWakeSession();
          ctx.transitionTo("sleep");
        }
        return;
      }
      ctx.transitionTo("sleep");
    });
  },
  answer: (ctx: ChatFlowContext) => {
    setFace("answering");
    ctx.enterMusicAfterAnswer = false;
    ctx.musicDisplayText = "";
    display({
      status: "answering...",
      RGB: "#00c8a3",
    });
    const currentAnswerId = ctx.answerId;
    if (isImMode) {
      const prompt: {
        role: "system" | "user";
        content: string;
      }[] = [
          {
            role: "user",
            content: ctx.asrText,
          },
        ];
      sendWhisplayIMMessage(prompt)
        .then((ok) => {
          if (ok) {
            display({
              status: "idle",
              emoji: "😊",
              RGB: "#000055",
              image_icon_visible: false,
            });
          } else {
            display({
              status: "error",
              emoji: "⚠️",
              text: "OpenClaw send failed",
              image_icon_visible: false,
            });
          }
        })
        .finally(() => {
          clearPendingCapturedImgForChat();
          ctx.transitionTo("sleep");
        });
      return;
    }
    onButtonPressed(() => {
      ctx.transitionTo("listening");
    });
    onButtonReleased(noop);
    const {
      partial,
      endPartial,
      getPlayEndPromise,
      stop: stopPlaying,
    } = ctx.streamResponser;
    let llmResponseText = "";
    const trackingPartial = (text: string): void => {
      llmResponseText += text;
      if (currentAnswerId === ctx.answerId) partial(text);
    };
    ctx.partialThinking = "";
    ctx.thinkingSentences = [];
    [() => Promise.resolve().then(() => ""), getSystemPromptWithKnowledge]
    [enableRAG ? 1 : 0](ctx.asrText)
      .then((res: string) => {
        let knowledgePrompt = res;
        if (res) {
          console.log("Retrieved knowledge for RAG:\n", res);
        }
        if (ctx.knowledgePrompts.includes(res)) {
          console.log(
            "[RAG] Knowledge prompt already used in this session, skipping to avoid repetition.",
          );
          knowledgePrompt = "";
        }
        if (knowledgePrompt) {
          ctx.knowledgePrompts.push(knowledgePrompt);
        }
        display({
          rag_icon_visible: Boolean(enableRAG && knowledgePrompt),
        });
        const prompt: {
          role: "system" | "user";
          content: string;
        }[] = compact([
          knowledgePrompt
            ? {
              role: "system",
              content: knowledgePrompt,
            }
            : null,
          {
            role: "user",
            content: ctx.asrText,
          },
        ]);
        chatWithLLMStream(
          prompt,
          (text) => { if (currentAnswerId === ctx.answerId) trackingPartial(text); },
          () => currentAnswerId === ctx.answerId && endPartial(),
          (partialThinking) =>
            currentAnswerId === ctx.answerId &&
            ctx.partialThinkingCallback(partialThinking),
          (functionName: string, result?: string) => {
            if (
              functionName === "endConversation" &&
              result?.startsWith("[success]")
            ) {
              ctx.endAfterAnswer = true;
            }
            if (
              functionName === "generateImage" &&
              result?.startsWith("[success]")
            ) {
              const img = getLatestGenImg();
              if (img) {
                display({ image: img });
              }
            }
            if (
              functionName.startsWith("playMusic") &&
              result?.startsWith("[success]")
            ) {
              ctx.enterMusicAfterAnswer = true;
              ctx.musicDisplayText = result.replace(/^\[success\]/, "").trim();
            }
            if (result) {
              display({
                text: `[${functionName}]${result}`,
              });
            } else {
              display({
                text: `Invoking [${functionName}]... {count}s`,
              });
            }
          },
        );
      });
    getPlayEndPromise().then(() => {
      if (ctx.currentFlowName === "answer") {
        autoSaveExchange(ctx.asrText, llmResponseText);
        clearPendingCapturedImgForChat();
        display({ image_icon_visible: false });
        if (ctx.wakeSessionActive || ctx.endAfterAnswer) {
          if (ctx.endAfterAnswer) {
            ctx.endWakeSession();
            ctx.transitionTo("sleep");
          } else {
            ctx.transitionTo("wake_listening");
          }
          return;
        }
        if (ctx.enterMusicAfterAnswer) {
          ctx.transitionTo("music");
          return;
        }
        const img = getLatestDisplayImg();
        if (img) {
          ctx.transitionTo("image");
        } else {
          ctx.transitionTo("sleep");
        }
      }
    });
    onButtonPressed(() => {
      stopPlaying();
      clearPendingCapturedImgForChat();
      display({ image_icon_visible: false });
      ctx.transitionTo("listening");
    });
    onButtonReleased(noop);
  },
  image: (ctx: ChatFlowContext) => {
    onButtonPressed(() => {
      display({ image: "" });
      ctx.transitionTo("listening");
    });
    onButtonReleased(noop);
  },
  wake_log_listening: (ctx: ChatFlowContext) => {
    setFace("listening");
    ctx.answerId += 1;
    const recordFilePath = `${ctx.recordingsDir}/log-${Date.now()}.${recordFileFormat}`;
    ctx.currentRecordFilePath = recordFilePath;

    onButtonDoubleClick(null);
    onButtonPressed(() => {
      ctx.transitionTo("sleep");
    });
    onButtonReleased(noop);

    display({
      status: "listening",
      emoji: "",
      RGB: "#00ff00",
      text: "Listening...",
      rag_icon_visible: false,
    });

    getDynamicVoiceDetectLevel().then((level) => {
      recordAudio(recordFilePath, ctx.wakeRecordMaxSec, level)
        .then(() => {
          if (ctx.currentFlowName !== "wake_log_listening") return;
          setFace("buffering");
          display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: "Processing..." });
          ctx.transitionTo("log_processing");
        })
        .catch((err) => {
          console.error("[wake_log_listening] Recording error:", err);
          if (ctx.currentFlowName === "wake_log_listening") ctx.transitionTo("sleep");
        });
    });
  },
  log_listening: (ctx: ChatFlowContext) => {
    ctx.answerId += 1;
    const recordFilePath = `${ctx.recordingsDir}/log-${Date.now()}.${recordFileFormat}`;
    ctx.currentRecordFilePath = recordFilePath;

    onButtonDoubleClick(null);
    onButtonPressed(noop);

    if (!isButtonDown()) {
      ctx.transitionTo("sleep");
      return;
    }

    const { result, stop } = recordAudioManually(recordFilePath);

    onButtonReleased(() => {
      onButtonReleased(noop);
      stop();
      setFace("buffering");
      display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: "Processing..." });
    });

    display({
      status: "listening",
      emoji: "",
      RGB: "#00ff00",
      text: "Listening...",
      rag_icon_visible: false,
    });

    result
      .then(() => {
        if (ctx.currentFlowName !== "log_listening") return;
        ctx.transitionTo("log_processing");
      })
      .catch((err) => {
        console.error("[log_listening] Recording error:", err);
        ctx.transitionTo("sleep");
      });
  },
  log_processing: (ctx: ChatFlowContext) => {
    ctx.streamResponser.stop();
    onButtonPressed(() => { ctx.transitionTo("sleep"); });
    onButtonReleased(noop);
    display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: "Processing..." });

    ctx.logLastDynamicFollowup = "";
    ctx.logLogType = "TASK";
    ctx.logDynamicFollowupCount = 0;

    const recordFilePath = ctx.currentRecordFilePath;
    const startTime = Date.now();

    saveLogEntry({ audioPath: recordFilePath, timestamp: startTime, type: "log" })
      .then(async (transcript) => {
        if (ctx.currentFlowName !== "log_processing") return;
        display({ last_log_at: startTime });
        ctx.logInitialTranscript = transcript;
        console.log("[log_processing] transcript:", transcript);
        const { question: dynamicQuestion, logType } = await generateDynamicFollowup(
          transcript, "", ""
        );
        console.log("[log_processing] dynamicQuestion:", dynamicQuestion, "logType:", logType);
        if (ctx.currentFlowName !== "log_processing") return;
        ctx.logLogType = logType;
        if (dynamicQuestion) {
          ctx.logDynamicFollowupCount = 1;
          ctx.logLastDynamicFollowup = dynamicQuestion;
          ctx.pendingLogResponseText = dynamicQuestion;
          ctx.transitionTo("log_dynamic_followup_response");
        } else {
          ctx.pendingLogResponseText = FOLLOWUP_1[logType];
          ctx.transitionTo("log_response");
        }
      })
      .catch((err) => {
        console.error("[log_processing] Error:", err);
        if (ctx.currentFlowName === "log_processing") {
          ctx.pendingLogResponseText = FOLLOWUP_1[ctx.logLogType];
          ctx.transitionTo("log_response");
        }
      });
  },
  log_dynamic_followup_response: (ctx: ChatFlowContext) => {
    ctx.streamResponser.stop();
    setFace("answering");
    const question = ctx.pendingLogResponseText;
    ctx.pendingLogResponseText = "";
    display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: question });
    let longPressTimer: NodeJS.Timeout | null = null;
    onButtonPressed(() => {
      longPressTimer = setTimeout(() => {
        longPressTimer = null;
        if (ctx.currentFlowName !== "log_dynamic_followup_response") return;
        ctx.streamResponser.stop();
        ctx.transitionTo("log_dynamic_followup_listening");
      }, LONG_PRESS_MS);
    });
    onButtonReleased(() => {
      if (longPressTimer) {
        clearTimeout(longPressTimer);
        longPressTimer = null;
        ctx.streamResponser.stop();
        ctx.transitionTo("log_confirmation");
      }
    });
    void ctx.streamExternalReply(question);
    ctx.streamResponser.getPlayEndPromise().then(() => {
      if (ctx.currentFlowName !== "log_dynamic_followup_response") return;
      if (longPressTimer) { clearTimeout(longPressTimer); longPressTimer = null; }
      if (isButtonDown()) {
        ctx.transitionTo("log_dynamic_followup_listening");
      } else {
        ctx.transitionTo("log_dynamic_followup_wait");
      }
    });
  },
  log_dynamic_followup_wait: (ctx: ChatFlowContext) => {
    setFace("idle");
    onButtonDoubleClick(null);
    display({ status: "idle", emoji: "", RGB: "#000033", text: "Hold to answer...", rag_icon_visible: false });
    let longPressTimer: NodeJS.Timeout | null = null;
    onButtonPressed(() => {
      longPressTimer = setTimeout(() => {
        longPressTimer = null;
        if (ctx.currentFlowName !== "log_dynamic_followup_wait") return;
        ctx.transitionTo("log_dynamic_followup_listening");
      }, LONG_PRESS_MS);
    });
    onButtonReleased(() => {
      if (longPressTimer) {
        clearTimeout(longPressTimer);
        longPressTimer = null;
      }
      ctx.transitionTo("log_confirmation");
    });
    setTimeout(() => {
      if (ctx.currentFlowName === "log_dynamic_followup_wait") {
        ctx.transitionTo("sleep");
      }
    }, FOLLOWUP_WAIT_TIMEOUT_MS);
  },
  log_dynamic_followup_listening: (ctx: ChatFlowContext) => {
    ctx.answerId += 1;
    const recordFilePath = `${ctx.recordingsDir}/log-dynamic-${Date.now()}.${recordFileFormat}`;
    ctx.currentRecordFilePath = recordFilePath;

    onButtonDoubleClick(null);
    onButtonPressed(noop);

    if (!isButtonDown()) {
      ctx.transitionTo("log_dynamic_followup_wait");
      return;
    }

    setFace("listening");

    const { result, stop } = recordAudioManually(recordFilePath);

    onButtonReleased(() => {
      onButtonReleased(noop);
      stop();
      setFace("buffering");
      display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: "Processing..." });
    });

    display({ status: "listening", emoji: "", RGB: "#00ff00", text: "Listening...", rag_icon_visible: false });

    result
      .then(async () => {
        if (ctx.currentFlowName !== "log_dynamic_followup_listening") return;
        const dynamicResponse = await saveLogEntry({ audioPath: recordFilePath, timestamp: Date.now(), type: "followup", log_type: ctx.logLogType, question: ctx.logLastDynamicFollowup });
        if (ctx.currentFlowName !== "log_dynamic_followup_listening") return;
        // Attempt a second dynamic follow-up (cap at 2 total)
        if (ctx.logDynamicFollowupCount < 2) {
          const { question: secondQuestion } = await generateDynamicFollowup(
            ctx.logInitialTranscript,
            ctx.logLastDynamicFollowup,
            dynamicResponse
          );
          if (ctx.currentFlowName !== "log_dynamic_followup_listening") return;
          if (secondQuestion) {
            ctx.logDynamicFollowupCount = 2;
            ctx.logLastDynamicFollowup = secondQuestion;
            ctx.pendingLogResponseText = secondQuestion;
            ctx.transitionTo("log_dynamic_followup_response");
            return;
          }
        }
        ctx.pendingLogResponseText = FOLLOWUP_1_WITH_TRANSITION[ctx.logLogType];
        ctx.transitionTo("log_response");
      })
      .catch((err) => {
        console.error("[log_dynamic_followup_listening] Error:", err);
        ctx.transitionTo("sleep");
      });
  },
  log_response: (ctx: ChatFlowContext) => {
    ctx.streamResponser.stop();
    setFace("answering");
    const followup1Text = ctx.pendingLogResponseText || FOLLOWUP_1[ctx.logLogType];
    ctx.pendingLogResponseText = "";
    display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: followup1Text });
    let longPressTimer: NodeJS.Timeout | null = null;
    onButtonPressed(() => {
      longPressTimer = setTimeout(() => {
        longPressTimer = null;
        if (ctx.currentFlowName !== "log_response") return;
        ctx.streamResponser.stop();
        ctx.transitionTo("log_followup_listening");
      }, LONG_PRESS_MS);
    });
    onButtonReleased(() => {
      if (longPressTimer) {
        clearTimeout(longPressTimer);
        longPressTimer = null;
        ctx.streamResponser.stop();
        ctx.transitionTo("log_confirmation");
      }
    });
    void ctx.streamExternalReply(followup1Text);
    ctx.streamResponser.getPlayEndPromise().then(() => {
      if (ctx.currentFlowName !== "log_response") return;
      if (longPressTimer) { clearTimeout(longPressTimer); longPressTimer = null; }
      if (isButtonDown()) {
        ctx.transitionTo("log_followup_listening");
      } else {
        ctx.transitionTo("log_followup_wait");
      }
    });
  },
  log_followup_wait: (ctx: ChatFlowContext) => {
    setFace("idle");
    onButtonDoubleClick(null);

    display({
      status: "idle",
      emoji: "",
      RGB: "#000033",
      text: "Hold to answer...",
      rag_icon_visible: false,
    });

    let longPressTimer: NodeJS.Timeout | null = null;
    onButtonPressed(() => {
      longPressTimer = setTimeout(() => {
        longPressTimer = null;
        if (ctx.currentFlowName !== "log_followup_wait") return;
        ctx.transitionTo("log_followup_listening");
      }, LONG_PRESS_MS);
    });

    onButtonReleased(() => {
      if (longPressTimer) {
        clearTimeout(longPressTimer);
        longPressTimer = null;
      }
      ctx.transitionTo("log_confirmation");
    });

    setTimeout(() => {
      if (ctx.currentFlowName === "log_followup_wait") {
        ctx.transitionTo("sleep");
      }
    }, FOLLOWUP_WAIT_TIMEOUT_MS);
  },
  log_followup_listening: (ctx: ChatFlowContext) => {
    ctx.answerId += 1;
    const recordFilePath = `${ctx.recordingsDir}/log-followup-${Date.now()}.${recordFileFormat}`;
    ctx.currentRecordFilePath = recordFilePath;

    onButtonDoubleClick(null);
    onButtonPressed(noop);

    if (!isButtonDown()) {
      ctx.transitionTo("log_followup_wait");
      return;
    }

    setFace("listening");

    const { result, stop } = recordAudioManually(recordFilePath);

    onButtonReleased(() => {
      onButtonReleased(noop);
      stop();
      setFace("buffering");
      display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: "Processing..." });
    });

    display({
      status: "listening",
      emoji: "",
      RGB: "#00ff00",
      text: "Listening...",
      rag_icon_visible: false,
    });

    result
      .then(() => {
        if (ctx.currentFlowName !== "log_followup_listening") return;
        saveLogEntry({ audioPath: recordFilePath, timestamp: Date.now(), type: "followup", log_type: ctx.logLogType, question: FOLLOWUP_1_WITH_TRANSITION[ctx.logLogType] });
        ctx.pendingLogResponseText = FOLLOWUP_2[ctx.logLogType];
        ctx.transitionTo("log_followup_response");
      })
      .catch((err) => {
        console.error("[log_followup_listening] Recording error:", err);
        ctx.transitionTo("sleep");
      });
  },
  log_followup_response: (ctx: ChatFlowContext) => {
    ctx.streamResponser.stop();
    setFace("answering");
    const followup2Text = ctx.pendingLogResponseText || FOLLOWUP_2[ctx.logLogType];
    ctx.pendingLogResponseText = "";
    display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: followup2Text });
    let longPressTimer: NodeJS.Timeout | null = null;
    onButtonPressed(() => {
      longPressTimer = setTimeout(() => {
        longPressTimer = null;
        if (ctx.currentFlowName !== "log_followup_response") return;
        ctx.streamResponser.stop();
        ctx.transitionTo("log_followup_2_listening");
      }, LONG_PRESS_MS);
    });
    onButtonReleased(() => {
      if (longPressTimer) {
        clearTimeout(longPressTimer);
        longPressTimer = null;
        ctx.streamResponser.stop();
        ctx.transitionTo("log_confirmation");
      }
    });
    void ctx.streamExternalReply(followup2Text);
    ctx.streamResponser.getPlayEndPromise().then(() => {
      if (ctx.currentFlowName !== "log_followup_response") return;
      if (longPressTimer) { clearTimeout(longPressTimer); longPressTimer = null; }
      if (isButtonDown()) {
        ctx.transitionTo("log_followup_2_listening");
      } else {
        ctx.transitionTo("log_followup_2_wait");
      }
    });
  },
  log_followup_2_wait: (ctx: ChatFlowContext) => {
    setFace("idle");
    onButtonDoubleClick(null);

    display({
      status: "idle",
      emoji: "",
      RGB: "#000033",
      text: "Hold to answer...",
      rag_icon_visible: false,
    });

    let longPressTimer: NodeJS.Timeout | null = null;
    onButtonPressed(() => {
      longPressTimer = setTimeout(() => {
        longPressTimer = null;
        if (ctx.currentFlowName !== "log_followup_2_wait") return;
        ctx.transitionTo("log_followup_2_listening");
      }, LONG_PRESS_MS);
    });

    onButtonReleased(() => {
      if (longPressTimer) {
        clearTimeout(longPressTimer);
        longPressTimer = null;
      }
      ctx.transitionTo("log_confirmation");
    });

    setTimeout(() => {
      if (ctx.currentFlowName === "log_followup_2_wait") {
        ctx.transitionTo("log_confirmation");
      }
    }, FOLLOWUP_WAIT_TIMEOUT_MS);
  },
  log_followup_2_listening: (ctx: ChatFlowContext) => {
    ctx.answerId += 1;
    const recordFilePath = `${ctx.recordingsDir}/log-followup2-${Date.now()}.${recordFileFormat}`;
    ctx.currentRecordFilePath = recordFilePath;

    onButtonDoubleClick(null);
    onButtonPressed(noop);

    if (!isButtonDown()) {
      ctx.transitionTo("log_followup_2_wait");
      return;
    }

    setFace("listening");

    const { result, stop } = recordAudioManually(recordFilePath);

    onButtonReleased(() => {
      onButtonReleased(noop);
      stop();
      setFace("buffering");
      display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: "Processing..." });
    });

    display({
      status: "listening",
      emoji: "",
      RGB: "#00ff00",
      text: "Listening...",
      rag_icon_visible: false,
    });

    result
      .then(() => {
        if (ctx.currentFlowName !== "log_followup_2_listening") return;
        saveLogEntry({ audioPath: recordFilePath, timestamp: Date.now(), type: "followup", log_type: ctx.logLogType, question: FOLLOWUP_2[ctx.logLogType] });
        ctx.transitionTo("log_confirmation");
      })
      .catch((err) => {
        console.error("[log_followup_2_listening] Recording error:", err);
        ctx.transitionTo("log_confirmation");
      });
  },
  log_confirmation: (ctx: ChatFlowContext) => {
    ctx.streamResponser.stop();
    setFace("answering");
    ctx.pendingLogResponseText = "";

    display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: LOG_CONFIRMATION });

    onButtonPressed(() => {
      ctx.streamResponser.stop();
      ctx.transitionTo("sleep");
    });
    onButtonReleased(noop);

    void ctx.streamExternalReply(LOG_CONFIRMATION);

    ctx.streamResponser.getPlayEndPromise().then(() => {
      if (ctx.currentFlowName === "log_confirmation") {
        ctx.transitionTo("sleep");
      }
    });
  },
  // ── End-of-day interactive flow ──────────────────────────────────────────
  eod_prompt: (ctx: ChatFlowContext) => {
    setFace("answering");
    display({
      status: "quick question",
      emoji: "",
      RGB: "#ff9900",
      text: EOD_QUESTION,
    });

    onButtonPressed(() => {
      ctx.streamResponser.stop();
      ctx.transitionTo("sleep");
    });
    onButtonReleased(noop);

    void ctx.streamExternalReply(EOD_QUESTION);

    ctx.streamResponser.getPlayEndPromise().then(() => {
      if (ctx.currentFlowName === "eod_prompt") {
        ctx.transitionTo("eod_wait");
      }
    });
  },

  eod_wait: (ctx: ChatFlowContext) => {
    setFace("idle");
    onButtonDoubleClick(null);

    display({
      status: "idle",
      emoji: "",
      RGB: "#331a00",
      text: "Hold to respond, or press briefly to skip.",
      rag_icon_visible: false,
    });

    let skipTimer: NodeJS.Timeout | null = null;

    onButtonPressed(() => {
      skipTimer = setTimeout(() => {
        skipTimer = null;
        setFace("listening");
        display({ status: "listening", emoji: "", RGB: "#00ff00", text: "Listening...", rag_icon_visible: false });
        ctx.transitionTo("eod_listening");
      }, 500);
    });

    onButtonReleased(() => {
      if (skipTimer) {
        clearTimeout(skipTimer);
        skipTimer = null;
        // Short press = skip → go straight to confirmation
        ctx.transitionTo("eod_confirmation");
      }
    });

    setTimeout(() => {
      if (ctx.currentFlowName === "eod_wait") {
        ctx.transitionTo("sleep");
      }
    }, FOLLOWUP_WAIT_TIMEOUT_MS);
  },

  eod_listening: (ctx: ChatFlowContext) => {
    ctx.answerId += 1;
    ctx.logLogType = "TASK";
    ctx.logLastDynamicFollowup = "";
    ctx.logDynamicFollowupCount = 0;
    const recordFilePath = `${ctx.recordingsDir}/eod-${Date.now()}.${recordFileFormat}`;
    ctx.currentRecordFilePath = recordFilePath;

    onButtonDoubleClick(null);
    onButtonPressed(noop);

    if (!isButtonDown()) {
      ctx.transitionTo("eod_wait");
      return;
    }

    setFace("listening");

    const { result, stop } = recordAudioManually(recordFilePath);

    onButtonReleased(() => {
      stop();
      setFace("buffering");
      display({ status: "quick question", emoji: "", RGB: "#ff9900", text: "Processing..." });
    });

    display({
      status: "listening",
      emoji: "",
      RGB: "#00ff00",
      text: "Listening...",
      rag_icon_visible: false,
    });

    result.then(async () => {
      if (ctx.currentFlowName !== "eod_listening") return;
      const transcript = await saveLogEntry({ audioPath: recordFilePath, timestamp: Date.now(), type: "eod" });
      if (ctx.currentFlowName !== "eod_listening") return;
      const { question: dynamicQuestion, logType } = await generateDynamicFollowup(transcript, "", "", true);
      if (ctx.currentFlowName !== "eod_listening") return;
      ctx.logLogType = logType;
      if (dynamicQuestion) {
        ctx.logLastDynamicFollowup = dynamicQuestion;
        ctx.pendingLogResponseText = dynamicQuestion;
        ctx.transitionTo("eod_dynamic_followup_response");
      } else {
        ctx.transitionTo("eod_followup_1");
      }
    }).catch(() => ctx.transitionTo("sleep"));
  },

  eod_dynamic_followup_response: (ctx: ChatFlowContext) => {
    ctx.streamResponser.stop();
    setFace("answering");
    const question = ctx.pendingLogResponseText;
    ctx.pendingLogResponseText = "";
    display({ status: "quick question", emoji: "", RGB: "#ff9900", text: question });
    onButtonPressed(noop);
    onButtonReleased(noop);
    void ctx.streamExternalReply(question);
    ctx.streamResponser.getPlayEndPromise().then(() => {
      if (ctx.currentFlowName === "eod_dynamic_followup_response") {
        ctx.transitionTo("eod_dynamic_followup_wait");
      }
    });
  },

  eod_dynamic_followup_wait: (ctx: ChatFlowContext) => {
    setFace("idle");
    onButtonDoubleClick(null);
    display({ status: "idle", emoji: "", RGB: "#331a00", text: "Hold to respond, or press briefly to skip.", rag_icon_visible: false });

    let skipTimer: NodeJS.Timeout | null = null;

    onButtonPressed(() => {
      skipTimer = setTimeout(() => {
        skipTimer = null;
        setFace("listening");
        display({ status: "listening", emoji: "", RGB: "#00ff00", text: "Listening...", rag_icon_visible: false });
        ctx.transitionTo("eod_dynamic_followup_listening");
      }, 500);
    });

    onButtonReleased(() => {
      if (skipTimer) {
        clearTimeout(skipTimer);
        skipTimer = null;
        ctx.transitionTo("eod_followup_1");
      }
    });

    setTimeout(() => {
      if (ctx.currentFlowName === "eod_dynamic_followup_wait") {
        ctx.transitionTo("eod_followup_1");
      }
    }, FOLLOWUP_WAIT_TIMEOUT_MS);
  },

  eod_dynamic_followup_listening: (ctx: ChatFlowContext) => {
    ctx.answerId += 1;
    const recordFilePath = `${ctx.recordingsDir}/eod-dynamic-${Date.now()}.${recordFileFormat}`;
    ctx.currentRecordFilePath = recordFilePath;

    onButtonDoubleClick(null);
    onButtonPressed(noop);

    if (!isButtonDown()) {
      ctx.transitionTo("eod_dynamic_followup_wait");
      return;
    }

    setFace("listening");

    const { result, stop } = recordAudioManually(recordFilePath);

    onButtonReleased(() => {
      stop();
      setFace("buffering");
      display({ status: "quick question", emoji: "", RGB: "#ff9900", text: "Processing..." });
    });

    display({ status: "listening", emoji: "", RGB: "#00ff00", text: "Listening...", rag_icon_visible: false });

    result.then(async () => {
      if (ctx.currentFlowName !== "eod_dynamic_followup_listening") return;
      await saveLogEntry({ audioPath: recordFilePath, timestamp: Date.now(), type: "eod", question: ctx.logLastDynamicFollowup });
      if (ctx.currentFlowName !== "eod_dynamic_followup_listening") return;
      ctx.transitionTo("eod_followup_1");
    }).catch(() => ctx.transitionTo("sleep"));
  },

  eod_followup_1: (ctx: ChatFlowContext) => {
    setFace("answering");
    display({ status: "quick question", emoji: "", RGB: "#ff9900", text: EOD_FOLLOWUP_1 });

    onButtonPressed(() => {
      ctx.streamResponser.stop();
      ctx.transitionTo("sleep");
    });
    onButtonReleased(noop);

    void ctx.streamExternalReply(EOD_FOLLOWUP_1);

    ctx.streamResponser.getPlayEndPromise().then(() => {
      if (ctx.currentFlowName === "eod_followup_1") {
        ctx.transitionTo("eod_followup_wait");
      }
    });
  },

  eod_followup_wait: (ctx: ChatFlowContext) => {
    setFace("idle");
    onButtonDoubleClick(null);

    display({
      status: "idle",
      emoji: "",
      RGB: "#331a00",
      text: "Hold to respond, or press briefly to skip.",
      rag_icon_visible: false,
    });

    let skipTimer: NodeJS.Timeout | null = null;

    onButtonPressed(() => {
      skipTimer = setTimeout(() => {
        skipTimer = null;
        setFace("listening");
        display({ status: "listening", emoji: "", RGB: "#00ff00", text: "Listening...", rag_icon_visible: false });
        ctx.transitionTo("eod_followup_listening");
      }, 500);
    });

    onButtonReleased(() => {
      if (skipTimer) {
        clearTimeout(skipTimer);
        skipTimer = null;
        ctx.transitionTo("eod_confirmation");
      }
    });

    setTimeout(() => {
      if (ctx.currentFlowName === "eod_followup_wait") {
        ctx.transitionTo("eod_confirmation");
      }
    }, FOLLOWUP_WAIT_TIMEOUT_MS);
  },

  eod_followup_listening: (ctx: ChatFlowContext) => {
    ctx.answerId += 1;
    const recordFilePath = `${ctx.recordingsDir}/eod-followup-${Date.now()}.${recordFileFormat}`;
    ctx.currentRecordFilePath = recordFilePath;

    onButtonDoubleClick(null);
    onButtonPressed(noop);

    if (!isButtonDown()) {
      ctx.transitionTo("eod_followup_wait");
      return;
    }

    setFace("listening");

    const { result, stop } = recordAudioManually(recordFilePath);

    onButtonReleased(() => {
      stop();
      setFace("buffering");
      display({ status: "quick question", emoji: "", RGB: "#ff9900", text: "Processing..." });
    });

    display({
      status: "listening",
      emoji: "",
      RGB: "#00ff00",
      text: "Listening...",
      rag_icon_visible: false,
    });

    result.then(() => {
      if (ctx.currentFlowName !== "eod_followup_listening") return;
      saveLogEntry({ audioPath: recordFilePath, timestamp: Date.now(), type: "eod" });
      ctx.transitionTo("eod_confirmation");
    }).catch(() => ctx.transitionTo("sleep"));
  },

  eod_followup_2: (ctx: ChatFlowContext) => {
    setFace("answering");
    display({ status: "quick question", emoji: "", RGB: "#ff9900", text: EOD_FOLLOWUP_2 });

    onButtonPressed(() => {
      ctx.streamResponser.stop();
      ctx.transitionTo("sleep");
    });
    onButtonReleased(noop);

    void ctx.streamExternalReply(EOD_FOLLOWUP_2);

    ctx.streamResponser.getPlayEndPromise().then(() => {
      if (ctx.currentFlowName === "eod_followup_2") {
        ctx.transitionTo("eod_followup_2_wait");
      }
    });
  },

  eod_followup_2_wait: (ctx: ChatFlowContext) => {
    setFace("idle");
    onButtonDoubleClick(null);

    display({
      status: "idle",
      emoji: "",
      RGB: "#331a00",
      text: "Hold to respond, or press briefly to skip.",
      rag_icon_visible: false,
    });

    let skipTimer: NodeJS.Timeout | null = null;

    onButtonPressed(() => {
      skipTimer = setTimeout(() => {
        skipTimer = null;
        setFace("listening");
        display({ status: "listening", emoji: "", RGB: "#00ff00", text: "Listening...", rag_icon_visible: false });
        ctx.transitionTo("eod_followup_2_listening");
      }, 500);
    });

    onButtonReleased(() => {
      if (skipTimer) {
        clearTimeout(skipTimer);
        skipTimer = null;
        ctx.transitionTo("eod_confirmation");
      }
    });

    setTimeout(() => {
      if (ctx.currentFlowName === "eod_followup_2_wait") {
        ctx.transitionTo("eod_confirmation");
      }
    }, FOLLOWUP_WAIT_TIMEOUT_MS);
  },

  eod_followup_2_listening: (ctx: ChatFlowContext) => {
    ctx.answerId += 1;
    const recordFilePath = `${ctx.recordingsDir}/eod-followup2-${Date.now()}.${recordFileFormat}`;
    ctx.currentRecordFilePath = recordFilePath;

    onButtonDoubleClick(null);
    onButtonPressed(noop);

    if (!isButtonDown()) {
      ctx.transitionTo("eod_followup_2_wait");
      return;
    }

    setFace("listening");

    const { result, stop } = recordAudioManually(recordFilePath);

    onButtonReleased(() => {
      stop();
      setFace("buffering");
      display({ status: "quick question", emoji: "", RGB: "#ff9900", text: "Processing..." });
    });

    display({
      status: "listening",
      emoji: "",
      RGB: "#00ff00",
      text: "Listening...",
      rag_icon_visible: false,
    });

    result.then(() => {
      if (ctx.currentFlowName !== "eod_followup_2_listening") return;
      saveLogEntry({ audioPath: recordFilePath, timestamp: Date.now(), type: "eod" });
      ctx.transitionTo("eod_confirmation");
    }).catch(() => ctx.transitionTo("sleep"));
  },

  eod_confirmation: (ctx: ChatFlowContext) => {
    setFace("answering");
    display({ status: "quick question", emoji: "", RGB: "#ff9900", text: EOD_CONFIRMATION });

    onButtonPressed(() => {
      ctx.streamResponser.stop();
      ctx.transitionTo("sleep");
    });
    onButtonReleased(noop);

    void ctx.streamExternalReply(EOD_CONFIRMATION);

    ctx.streamResponser.getPlayEndPromise().then(() => {
      if (ctx.currentFlowName === "eod_confirmation") {
        ctx.transitionTo("sleep");
      }
    });
  },
  external_answer: (ctx: ChatFlowContext) => {
    if (!ctx.pendingExternalReply && !ctx.pendingExternalImageUrl) {
      ctx.transitionTo("sleep");
      return;
    }
    display({
      status: "answering...",
      RGB: "#00c8a3",
      ...(ctx.pendingExternalEmoji ? { emoji: ctx.pendingExternalEmoji } : {}),
    });
    onButtonPressed(() => {
      ctx.streamResponser.stop();
      display({ image: "" });
      ctx.transitionTo("listening");
    });
    onButtonReleased(noop);
    const replyText = ctx.pendingExternalReply;
    const replyEmoji = ctx.pendingExternalEmoji;
    const replyImageUrl = ctx.pendingExternalImageUrl;
    ctx.currentExternalEmoji = replyEmoji;
    ctx.pendingExternalReply = "";
    ctx.pendingExternalEmoji = "";
    ctx.pendingExternalImageUrl = "";

    // Display the image if one was provided
    if (replyImageUrl) {
      display({ image: replyImageUrl });
    }

    if (replyText) {
      void ctx.streamExternalReply(replyText, replyEmoji);
      ctx.streamResponser.getPlayEndPromise().then(() => {
        if (ctx.currentFlowName !== "external_answer") return;
        if (ctx.wakeSessionActive || ctx.endAfterAnswer) {
          if (ctx.endAfterAnswer) {
            ctx.endWakeSession();
            ctx.transitionTo("sleep");
          } else {
            ctx.transitionTo("wake_listening");
          }
        } else if (replyImageUrl) {
          // Stay in image display mode after TTS finishes
          ctx.transitionTo("image");
        } else {
          ctx.transitionTo("sleep");
        }
      });
    } else {
      // Image only, no text to speak — go to image display mode
      ctx.transitionTo("image");
    }
  },
};
