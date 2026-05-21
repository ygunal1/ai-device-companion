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
  TASK: "How useful would it be for me to handle something like this, and why?",
  THINKING: "How useful would it be to have someone to think this through with, and why?",
  SOCIAL: "How useful would it be for me to handle something like this, and why?",
  REFLECTION: "How useful would it be for me to handle something like this, and why?",
};
const FOLLOWUP_1_WITH_TRANSITION: Record<LogType, string> = {
  TASK: "Got it. How useful would it be for me to handle something like this, and why?",
  THINKING: "Got it. How useful would it be to have someone to think this through with, and why?",
  SOCIAL: "Got it. How useful would it be for me to handle something like this, and why?",
  REFLECTION: "Got it. How useful would it be for me to handle something like this, and why?",
};
const FOLLOWUP_2: Record<LogType, string> = {
  TASK: "Are there any tools you would usually use for this?",
  THINKING: "Is this the kind of thing you'd normally talk through with a colleague or teammate?",
  SOCIAL: "If I could have responded in the moment — what would the most useful thing I could have said or done been?",
  REFLECTION: "If I could have responded in the moment — what would the most useful thing I could have said or done been?",
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

Your job is to generate ONE short follow-up question spoken aloud in natural
conversational language, or return null if enough context already exists.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — CLASSIFY THE LOG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Classify into one of four types:

TASK — concrete, delegatable, clear output (summarizing, scheduling,
finding, sending, reminding, formatting, notifying)

THINKING — open-ended, uncertain, or dialogic (decisions, tradeoffs,
being stuck, preparing for a conversation, reasoning through a problem,
reviewing an approach, architectural choices)

SOCIAL — navigating a relationship or interpersonal situation (conflict,
feedback, a difficult conversation, collaboration friction)

REFLECTION — retrospective rather than prospective (something that
already happened that the participant is processing or wants to
understand differently)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — TYPE-SPECIFIC LOGIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If THINKING → choose the best fit from:
- "What made this hard to think through on your own?"
- "Were you looking for information, a sounding board, or something else?"
- "What would a useful back-and-forth on this have looked like?"
- "What would the ideal next step have been if I could have responded?"

If SOCIAL → choose the best fit from:
- "What outcome were you hoping for in that situation?"
- "Were you looking for advice, a way to prepare, or something else?"
- "What made that difficult to navigate on your own?"

If REFLECTION → choose the best fit from:
- "What would have helped you work through that differently?"
- "What were you still unclear on when it came up?"
- "Is this something that keeps coming up, or was it specific to today?"

If TASK → apply priority rules below (first match wins)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — TASK PRIORITY RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Apply in order. First match wins. Return only the question or null.

1. Cut-off or incoherent log →
   "sorry, I didn't quite catch that — could you say that again?"
   [does not count as a dynamic follow-up]

2. Reminder or notification without timing →
   ask when and how often

3. Informational request without detail level →
   ask quick summary or detailed overview

4. Contact without method →
   ask how they'd want to reach out

5. Emotional expression or negative opinion →
   brief acknowledgement + "what's making it difficult?"

6. Noun-based log, usage or specifics unclear →
   ask about usage, attributes, or what specifically they needed

7. Verb-based log, scope or specifics missing →
   ask about timing, tools, or degree

8. Work context unclear AND agent could not infer it →
   ask independent/collaborative/meeting context

9. Recurring task or forgetting → null

10. Agent could infer missing details from context → null

11. Context check passes (what they were doing + what they wanted +
    people involved all clear) → null

12. Too short or vague →
    "can you tell me a bit more about what you had in mind?"

13. No call to action [FALLBACK] →
    "how can I help with that?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — STATIC FOLLOW-UPS (always after dynamic)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After 0–2 dynamic follow-ups, always ask the static follow-ups in order:

For TASK logs:
1. "Got it. How useful would it be for me to handle something like this,
   and why?"
2. "Are there any tools you would usually use for this?"
3. Confirmation: "Got it, I've noted that down."

FOR THINKING logs:
4. "How useful would it be to have someone to think this through with, and why?"
5. "Is this the kind of thing you'd normally talk through with a colleague or teammate?"
6. Confirmation: "Got it, I've noted that down."

For SOCIAL or REFLECTION logs:
5. "Got it. How useful would it be for me to handle something like this,
   and why?"
6. "If I could have responded in the moment — what would the most useful
   thing I could have said or done been?"
7. Confirmation: "Got it, I've noted that down."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARD LIMITS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Maximum 2 dynamic follow-ups per log
- No yes/no questions
- Maximum 15 words per question
- Return only the question string or null
- Never describe what you are doing — speak the question directly
- Never use filler: "Great!", "Sure!", "Absolutely!"
- After a dynamic follow-up is answered, re-evaluate full context
  including original log + all responses together before asking another

Return ONLY a JSON object on a single line — nothing else:
  {"dynamic_question": "<question or null>", "log_type": "<TASK|THINKING|SOCIAL|REFLECTION>"}

- dynamic_question: a single question in natural spoken language (max 15 words), or the JSON null value (not the string "null")
- log_type: the classification from STEP 1

Do not explain your reasoning. Do not return anything other than the JSON object.`;

async function generateDynamicFollowup(
  transcript: string,
  previousFollowup: string,
  previousResponse: string
): Promise<{ question: string | null; logType: LogType }> {
  const fallback = { question: null, logType: "TASK" as LogType };
  if (!openai) return fallback;
  const userContent = `Current log: "${transcript}"\nPrevious follow-up asked (if any): "${previousFollowup}"\nPrevious follow-up response (if any): "${previousResponse}"`;
  try {
    const completion = await openai.chat.completions.create({
      model: openaiLLMModel,
      messages: [
        { role: "system", content: DYNAMIC_FOLLOWUP_SYSTEM_PROMPT },
        { role: "user", content: userContent },
      ],
      max_tokens: 100,
      temperature: 0.7,
    });
    const raw = completion.choices[0]?.message?.content?.trim() || "{}";
    const parsed = JSON.parse(raw) as { dynamic_question?: string | null; log_type?: string };
    const logType: LogType =
      parsed.log_type === "THINKING" || parsed.log_type === "SOCIAL" || parsed.log_type === "REFLECTION"
        ? parsed.log_type
        : "TASK";
    let question = parsed.dynamic_question ?? null;
    if (question) {
      // Ensure only one question — take everything up to and including the first "?"
      const firstQ = question.indexOf("?");
      if (firstQ !== -1) question = question.slice(0, firstQ + 1).trim();
    }
    return { question: question || null, logType };
  } catch (err) {
    console.error("[DynamicFollowup] LLM call failed:", err);
    return fallback;
  }
}

const EOD_QUESTION = "Thinking about your day, is there anything you wish you could have used me for that you haven't logged?";
const EOD_FOLLOWUP_1 = "How useful would it be for me to handle something like this and why?";
const EOD_FOLLOWUP_2 = "Did you do anything about this when it came up today?";
const EOD_CONFIRMATION = "Got it, I've noted that down. Have a good evening.";

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
          setFace("answering");
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
      setFace("answering");
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

    const recordFilePath = ctx.currentRecordFilePath;
    const startTime = Date.now();

    saveLogEntry({ audioPath: recordFilePath, timestamp: startTime, type: "log" })
      .then(async (transcript) => {
        if (ctx.currentFlowName !== "log_processing") return;
        ctx.logInitialTranscript = transcript;
        console.log("[log_processing] transcript:", transcript);
        const { question: dynamicQuestion, logType } = await generateDynamicFollowup(transcript, "", "");
        console.log("[log_processing] dynamicQuestion:", dynamicQuestion, "logType:", logType);
        if (ctx.currentFlowName !== "log_processing") return;
        ctx.logLogType = logType;
        if (dynamicQuestion) {
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
    const question = ctx.pendingLogResponseText;
    ctx.pendingLogResponseText = "";
    display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: question });
    onButtonPressed(noop);
    onButtonReleased(noop);
    void ctx.streamExternalReply(question);
    ctx.streamResponser.getPlayEndPromise().then(() => {
      if (ctx.currentFlowName === "log_dynamic_followup_response") {
        ctx.transitionTo("log_dynamic_followup_wait");
      }
    });
  },
  log_dynamic_followup_wait: (ctx: ChatFlowContext) => {
    setFace("idle");
    onButtonDoubleClick(null);
    display({ status: "idle", emoji: "", RGB: "#000033", text: "Hold to answer...", rag_icon_visible: false });
    onButtonPressed(() => {
      setFace("listening");
      display({ status: "listening", emoji: "", RGB: "#00ff00", text: "Listening...", rag_icon_visible: false });
      ctx.transitionTo("log_dynamic_followup_listening");
    });
    onButtonReleased(noop);
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
      setFace("answering");
      display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: "Processing..." });
    });

    display({ status: "listening", emoji: "", RGB: "#00ff00", text: "Listening...", rag_icon_visible: false });

    result
      .then(async () => {
        if (ctx.currentFlowName !== "log_dynamic_followup_listening") return;
        await saveLogEntry({ audioPath: recordFilePath, timestamp: Date.now(), type: "followup", log_type: ctx.logLogType, question: ctx.logLastDynamicFollowup });
        if (ctx.currentFlowName !== "log_dynamic_followup_listening") return;
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
    const followup1Text = ctx.pendingLogResponseText || FOLLOWUP_1[ctx.logLogType];
    ctx.pendingLogResponseText = "";
    display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: followup1Text });
    onButtonPressed(() => {
      ctx.streamResponser.stop();
      ctx.transitionTo("sleep");
    });
    onButtonReleased(noop);
    void ctx.streamExternalReply(followup1Text);
    ctx.streamResponser.getPlayEndPromise().then(() => {
      if (ctx.currentFlowName === "log_response") {
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

    onButtonPressed(() => {
      setFace("listening");
      display({ status: "listening", emoji: "", RGB: "#00ff00", text: "Listening...", rag_icon_visible: false });
      ctx.transitionTo("log_followup_listening");
    });

    onButtonReleased(noop);

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
      setFace("answering");
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
    const followup2Text = ctx.pendingLogResponseText || FOLLOWUP_2[ctx.logLogType];
    ctx.pendingLogResponseText = "";
    display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: followup2Text });
    onButtonPressed(() => {
      ctx.streamResponser.stop();
      ctx.transitionTo("sleep");
    });
    onButtonReleased(noop);
    void ctx.streamExternalReply(followup2Text);
    ctx.streamResponser.getPlayEndPromise().then(() => {
      if (ctx.currentFlowName === "log_followup_response") {
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

    onButtonPressed(() => {
      setFace("listening");
      display({ status: "listening", emoji: "", RGB: "#00ff00", text: "Listening...", rag_icon_visible: false });
      ctx.transitionTo("log_followup_2_listening");
    });

    onButtonReleased(noop);

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
      setFace("answering");
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
      status: "quick question",
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
      setFace("answering");
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
      const { question: dynamicQuestion, logType } = await generateDynamicFollowup(transcript, "", "");
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
    display({ status: "quick question", emoji: "", RGB: "#331a00", text: "Hold to respond, or press briefly to skip.", rag_icon_visible: false });

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
      setFace("answering");
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
      status: "quick question",
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
      setFace("answering");
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
      ctx.transitionTo("eod_followup_2");
    }).catch(() => ctx.transitionTo("sleep"));
  },

  eod_followup_2: (ctx: ChatFlowContext) => {
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
      status: "quick question",
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
      setFace("answering");
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
