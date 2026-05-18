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

const FOLLOWUP_1 = "How useful would it be for me to handle something like this and why?";
const FOLLOWUP_1_WITH_TRANSITION = "Got it. How useful would it be for me to handle something like this and why?";
const FOLLOWUP_2 = "Are there any tools you would usually use for this?";
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

const DYNAMIC_FOLLOWUP_SYSTEM_PROMPT = `You are helping a researcher collect structured diary logs from knowledge workers.
A participant has just spoken a short voice log describing something they wished
an AI agent could help them with during their workday.

Your job is to ask ONE short follow-up question that would meaningfully improve
understanding of the situation, or return null if enough context exists.

CORE EVALUATION RULE:
Before generating any follow-up, ask yourself two questions:
1. Does a researcher already understand what the participant wanted and why?
2. Would an AI agent handling this task already know or be able to infer the
   missing information from the participant's environment, files, or context?

If the answer to either question is yes for all missing details, return null.
If a follow-up response has already been given, re-evaluate the original log
AND all previous responses together as a complete picture. Do not search for
new gaps introduced by the follow-up response itself. When the combined context
is sufficient, return null immediately.

NEVER ask about:
- Information an AI agent could infer from the participant's environment,
  open files, codebase, calendar, or active applications (e.g. programming
  language, platform, file type, meeting details already in calendar)
- Timing or urgency unless the participant explicitly mentioned a deadline
- Recurrence unless it is a reminder or notification request
- Tasks described as recurring or something the participant keeps forgetting
- Anything already present in the log or answered in a previous follow-up
- Restating or rephrasing what the participant already said in more detail

DO NOT ask yes/no questions.
DO NOT ask more than one question per turn.
When in doubt between asking and returning null, return null.

FOLLOW-UP LIMITS:
- Dynamic follow-ups (questions you generate): maximum 2 total across the
  entire interaction. After 2 dynamic follow-ups, return null regardless of
  remaining context gaps — the fixed follow-up questions will follow.
- Static follow-ups (the researcher's fixed questions asked after dynamic
  follow-ups are complete) are separate and always asked. Do not count them
  toward your limit.
- After each dynamic follow-up response, re-evaluate the full context.
  If sufficient, return null early rather than using both dynamic slots.

SPECIFIC RULES in order of priority:

UNCLEAR INTENT OR NO CALL TO ACTION
- If the participant's log is a statement, observation, or description
  without a clear request or action for an agent to take (e.g. "I have
  a meeting later", "my code isn't working", "I need to email someone"):
  ask how the agent can help with that.
- Example: "my standup is in 10 minutes" → "how can I help with that?"
- Example: "I have a lot of emails to get through" → "how can I help with that?"
- Do not apply this rule if the log already contains a clear implicit
  request (e.g. "remind me", "take notes", "summarize") even if no
  explicit call to action is stated.

VAGUE ACTION REQUESTS
- If the participant makes a clear request but provides no description of
  the specific problem, symptom, or desired outcome needed to act on it,
  ask what the issue or goal is.
- This applies even when an agent could access relevant files or context —
  the agent still needs to know what to look for or what went wrong.
- Example: "can you help me debug my code" → "what's the issue you're running into?"
- Example: "help me write an email" → "what's the email for?"
- Do not apply this rule if the problem or outcome is already clear from
  the log, even if briefly stated (e.g. "there's a null pointer error in
  this file", "help me reply to this client complaint").

REMINDERS AND NOTIFICATIONS
- If the participant says "remind me", "notify me", "let me know", or similar,
  and does not specify when or how often: ask when and how often they would
  want to be reminded or notified.
- Example: "remind me to check in with my team" → ask when and how often.

INFORMATIONAL REQUESTS
- If the participant asks for information (e.g. "tell me about", "what is",
  "explain") and it is unclear whether they want a quick answer or detailed
  overview: ask which they prefer.
- Do not apply this rule to action requests that happen to involve
  information (e.g. "summarize my emails").

TASK CONTEXT
- If the participant mentions a specific task but does not describe whether
  they are working independently, collaborating, or in a meeting: ask which.
  - If working independently and main goal is unclear: ask what their
    main goal is.
- If tools or systems are missing AND an AI agent could not infer them from
  context: ask what tools or systems are involved.

EMOTIONAL OR OPINION EXPRESSIONS
- If the participant expresses frustration or a negative opinion: ask what
  specifically made it difficult or frustrating.

VAGUE OR UNCLEAR LOGS
- If what the participant wanted done is genuinely unclear: ask what they
  had in mind.
- If the log is too short to extract any intent: ask "can you tell me a
  bit more about what you had in mind?"
- Do not ask for specificity when the general intent is already clear.

EXAMPLES OF WHEN TO RETURN NULL:
- "I have a bug in my code in this file" → null (file specified, agent can look)
- "There's a null pointer error on line 42" → null (symptom is clear)
- "Follow-up email to a client I keep forgetting" → null
- "Remind me to check in with my team" + response "we're working on text analytics" → null

EXAMPLES OF WHEN TO ASK:
- "Can you help me debug my code" → "what's the issue you're running into?"
- "Help me write an email" → "what's the email for?"
- "I need help with my presentation" → "what do you need done with it?"

Return ONLY one of the following:
- A single question of no more than 15 words in natural spoken language
- The exact string: null

Do not explain your reasoning. Do not return more than one question.`;

async function generateDynamicFollowup(
  transcript: string,
  previousFollowup: string,
  previousResponse: string
): Promise<string | null> {
  if (!openai) return null;
  const userContent = `Current log: "${transcript}"\nPrevious follow-up asked (if any): "${previousFollowup}"\nPrevious follow-up response (if any): "${previousResponse}"`;
  try {
    const completion = await openai.chat.completions.create({
      model: openaiLLMModel,
      messages: [
        { role: "system", content: DEVICE_PERSONALITY_PROMPT },
        { role: "system", content: DYNAMIC_FOLLOWUP_SYSTEM_PROMPT },
        { role: "user", content: userContent },
      ],
      max_tokens: 60,
      temperature: 0.7,
    });
    const raw = completion.choices[0]?.message?.content?.trim() || "null";
    if (raw === "null" || !raw) return null;
    // Strip surrounding quotes the LLM sometimes adds
    const result = raw.replace(/^["']|["']$/g, "").trim();
    if (!result) return null;
    return result;
  } catch (err) {
    console.error("[DynamicFollowup] LLM call failed:", err);
    return null;
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
    onButtonPressed(() => { ctx.transitionTo("sleep"); });
    onButtonReleased(noop);
    display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: "Processing..." });

    ctx.logDynamicFollowupCount = 0;
    ctx.logLastDynamicFollowup = "";
    ctx.logLastDynamicResponse = "";

    const recordFilePath = ctx.currentRecordFilePath;
    const startTime = Date.now();

    const fallbackToFixed = () => {
      if (ctx.currentFlowName !== "log_processing") return;
      ctx.pendingLogResponseText = FOLLOWUP_1;
      ctx.logTTSPreStarted = true;
      ctx.logPlayEndPromise = ctx.streamResponser.getPlayEndPromise();
      display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: FOLLOWUP_1 });
      void ctx.streamExternalReply(FOLLOWUP_1);
      ctx.transitionTo("log_response");
    };

    saveLogEntry({ audioPath: recordFilePath, timestamp: startTime, type: "log" })
      .then(async (transcript) => {
        if (ctx.currentFlowName !== "log_processing") return;
        ctx.logInitialTranscript = transcript;
        console.log("[log_processing] transcript:", transcript);
        const dynamicQuestion = await generateDynamicFollowup(transcript, "", "");
        console.log("[log_processing] dynamicQuestion:", dynamicQuestion);
        if (ctx.currentFlowName !== "log_processing") return;
        if (dynamicQuestion) {
          ctx.logDynamicFollowupCount = 1;
          ctx.logLastDynamicFollowup = dynamicQuestion;
          ctx.pendingLogResponseText = dynamicQuestion;
          ctx.logTTSPreStarted = true;
          ctx.logPlayEndPromise = ctx.streamResponser.getPlayEndPromise();
          display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: dynamicQuestion });
          void ctx.streamExternalReply(dynamicQuestion);
          ctx.transitionTo("log_dynamic_followup_response");
        } else {
          fallbackToFixed();
        }
      })
      .catch((err) => {
        console.error("[log_processing] Error:", err);
        fallbackToFixed();
      });
  },
  log_dynamic_followup_response: (ctx: ChatFlowContext) => {
    const playEnd = ctx.logTTSPreStarted && ctx.logPlayEndPromise
      ? ctx.logPlayEndPromise
      : ctx.streamResponser.getPlayEndPromise();

    const question = ctx.pendingLogResponseText;
    display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: question });

    onButtonPressed(noop);
    onButtonReleased(noop);

    if (!ctx.logTTSPreStarted) {
      void ctx.streamExternalReply(question);
    }
    ctx.logTTSPreStarted = false;
    ctx.logPlayEndPromise = null;
    ctx.pendingLogResponseText = "";

    playEnd.then(() => {
      if (ctx.currentFlowName === "log_dynamic_followup_response") {
        ctx.transitionTo("log_dynamic_followup_wait");
      }
    });
  },
  log_dynamic_followup_wait: (ctx: ChatFlowContext) => {
    setFace("idle");
    onButtonDoubleClick(null);
    display({ status: "idle", emoji: "", RGB: "#000033", text: "Hold to answer...", rag_icon_visible: false });
    onButtonPressed(() => { ctx.transitionTo("log_dynamic_followup_listening"); });
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

    const { result, stop } = recordAudioManually(recordFilePath);

    onButtonReleased(() => {
      onButtonReleased(noop);
      stop();
      setFace("answering");
      ctx.pendingLogResponseText = FOLLOWUP_1_WITH_TRANSITION;
      ctx.logTTSPreStarted = true;
      ctx.logPlayEndPromise = ctx.streamResponser.getPlayEndPromise();
      display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: FOLLOWUP_1_WITH_TRANSITION });
      void ctx.streamExternalReply(FOLLOWUP_1_WITH_TRANSITION);
    });

    display({ status: "listening", emoji: "", RGB: "#00ff00", text: "Listening...", rag_icon_visible: false });

    result
      .then(() => {
        if (ctx.currentFlowName !== "log_dynamic_followup_listening") return;
        saveLogEntry({ audioPath: recordFilePath, timestamp: Date.now(), type: "followup", question: ctx.logLastDynamicFollowup });
        ctx.transitionTo("log_response");
      })
      .catch((err) => {
        console.error("[log_dynamic_followup_listening] Error:", err);
        ctx.transitionTo("sleep");
      });
  },
  log_response: (ctx: ChatFlowContext) => {
    const playEnd = ctx.logTTSPreStarted && ctx.logPlayEndPromise
      ? ctx.logPlayEndPromise
      : ctx.streamResponser.getPlayEndPromise();

    const followup1Text = ctx.pendingLogResponseText || FOLLOWUP_1;
    display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: followup1Text });

    onButtonPressed(() => {
      ctx.streamResponser.stop();
      ctx.transitionTo("sleep");
    });
    onButtonReleased(noop);

    if (!ctx.logTTSPreStarted) {
      void ctx.streamExternalReply(followup1Text);
    }
    ctx.logTTSPreStarted = false;
    ctx.logPlayEndPromise = null;
    ctx.pendingLogResponseText = "";

    playEnd.then(() => {
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

    const { result, stop } = recordAudioManually(recordFilePath);

    onButtonReleased(() => {
      onButtonReleased(noop);
      stop();
      setFace("answering");
      ctx.pendingLogResponseText = FOLLOWUP_2;
      ctx.logTTSPreStarted = true;
      ctx.logPlayEndPromise = ctx.streamResponser.getPlayEndPromise();
      display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: FOLLOWUP_2 });
      void ctx.streamExternalReply(FOLLOWUP_2);
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
        saveLogEntry({ audioPath: recordFilePath, timestamp: Date.now(), type: "followup", question: FOLLOWUP_1_WITH_TRANSITION });
        ctx.transitionTo("log_followup_response");
      })
      .catch((err) => {
        console.error("[log_followup_listening] Recording error:", err);
        ctx.transitionTo("sleep");
      });
  },
  log_followup_response: (ctx: ChatFlowContext) => {
    const playEnd = ctx.logTTSPreStarted && ctx.logPlayEndPromise
      ? ctx.logPlayEndPromise
      : ctx.streamResponser.getPlayEndPromise();

    display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: FOLLOWUP_2 });

    onButtonPressed(() => {
      ctx.streamResponser.stop();
      ctx.transitionTo("sleep");
    });
    onButtonReleased(noop);

    if (!ctx.logTTSPreStarted) {
      void ctx.streamExternalReply(FOLLOWUP_2);
    }
    ctx.logTTSPreStarted = false;
    ctx.logPlayEndPromise = null;
    ctx.pendingLogResponseText = "";

    playEnd.then(() => {
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
      ctx.transitionTo("log_followup_2_listening");
    });

    onButtonReleased(noop);

    setTimeout(() => {
      if (ctx.currentFlowName === "log_followup_2_wait") {
        ctx.transitionTo("sleep");
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

    const { result, stop } = recordAudioManually(recordFilePath);

    onButtonReleased(() => {
      onButtonReleased(noop);
      stop();
      setFace("answering");
      ctx.pendingLogResponseText = LOG_CONFIRMATION;
      ctx.logTTSPreStarted = true;
      ctx.logPlayEndPromise = ctx.streamResponser.getPlayEndPromise();
      display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: LOG_CONFIRMATION });
      void ctx.streamExternalReply(LOG_CONFIRMATION);
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
        saveLogEntry({ audioPath: recordFilePath, timestamp: Date.now(), type: "followup", question: FOLLOWUP_2 });
        ctx.transitionTo("log_confirmation");
      })
      .catch((err) => {
        console.error("[log_followup_2_listening] Recording error:", err);
        ctx.transitionTo("sleep");
      });
  },
  log_confirmation: (ctx: ChatFlowContext) => {
    const playEnd = ctx.logTTSPreStarted && ctx.logPlayEndPromise
      ? ctx.logPlayEndPromise
      : ctx.streamResponser.getPlayEndPromise();

    display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: LOG_CONFIRMATION });

    onButtonPressed(() => {
      ctx.streamResponser.stop();
      ctx.transitionTo("sleep");
    });
    onButtonReleased(noop);

    if (!ctx.logTTSPreStarted) {
      void ctx.streamExternalReply(LOG_CONFIRMATION);
    }
    ctx.logTTSPreStarted = false;
    ctx.logPlayEndPromise = null;
    ctx.pendingLogResponseText = "";

    playEnd.then(() => {
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
    const recordFilePath = `${ctx.recordingsDir}/eod-${Date.now()}.${recordFileFormat}`;
    ctx.currentRecordFilePath = recordFilePath;

    onButtonDoubleClick(null);
    onButtonPressed(noop);

    if (!isButtonDown()) {
      ctx.transitionTo("eod_wait");
      return;
    }

    const { result, stop } = recordAudioManually(recordFilePath);

    onButtonReleased(() => {
      stop();
      display({ status: "quick question", emoji: "", RGB: "#ff9900", text: EOD_FOLLOWUP_1 });
    });

    display({
      status: "listening",
      emoji: "",
      RGB: "#00ff00",
      text: "Listening...",
      rag_icon_visible: false,
    });

    result.then(() => {
      if (ctx.currentFlowName !== "eod_listening") return;
      saveLogEntry({ audioPath: recordFilePath, timestamp: Date.now(), type: "eod" });
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

    const { result, stop } = recordAudioManually(recordFilePath);

    onButtonReleased(() => {
      stop();
      display({ status: "quick question", emoji: "", RGB: "#ff9900", text: EOD_FOLLOWUP_2 });
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

    const { result, stop } = recordAudioManually(recordFilePath);

    onButtonReleased(() => {
      stop();
      display({ status: "quick question", emoji: "", RGB: "#ff9900", text: EOD_CONFIRMATION });
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
