import moment from "moment";
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

const LONG_PRESS_MS = parseInt(process.env.LONG_PRESS_MS || "1500");
const FOLLOWUP_WAIT_TIMEOUT_MS = parseInt(process.env.FOLLOWUP_WAIT_TIMEOUT_MS || "30000");
const SLEEP_DISPLAY_TEXT = "Long press the button to log an entry.";

const FOLLOWUP_1 = "On a scale of 1-5, how useful would that be?";
const FOLLOWUP_2 = "Got it. What would you normally do to take care of this?";
const LOG_CONFIRMATION = "I've noted this down, thank you!";

export const flowStates: Record<FlowName, FlowStateHandler> = {
  sleep: (ctx: ChatFlowContext) => {
    let longPressTimer: NodeJS.Timeout | null = null;

    onCameraModeExit(null);
    onButtonDoubleClick(null);

    onButtonPressed(() => {
      resetCameraModeControl();
      stopMusicPlayback();
      display({ text: "Listening...", RGB: "#00aa44" });
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
        image: "",
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
      stop();
      // Pre-start TTS immediately on button release (parallel with recording finalization + ASR)
      ctx.pendingLogResponseText = FOLLOWUP_1;
      ctx.logTTSPreStarted = true;
      display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: FOLLOWUP_1 });
      void ctx.streamExternalReply(FOLLOWUP_1);
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
        saveLogEntry({ audioPath: recordFilePath, timestamp: Date.now(), type: "log" });
        ctx.transitionTo("log_response");
      })
      .catch((err) => {
        console.error("[log_listening] Recording error:", err);
        ctx.transitionTo("sleep");
      });
  },
  log_response: (ctx: ChatFlowContext) => {
    const fullText = ctx.logTTSPreStarted && ctx.pendingLogResponseText
      ? ctx.pendingLogResponseText
      : FOLLOWUP_1;

    display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: fullText });

    onButtonPressed(() => {
      ctx.streamResponser.stop();
      ctx.transitionTo("sleep");
    });
    onButtonReleased(noop);

    if (!ctx.logTTSPreStarted) {
      void ctx.streamExternalReply(fullText);
    }
    ctx.logTTSPreStarted = false;
    ctx.pendingLogResponseText = "";

    ctx.streamResponser.getPlayEndPromise().then(() => {
      if (ctx.currentFlowName === "log_response") {
        ctx.transitionTo("log_followup_wait");
      }
    });
  },
  log_followup_wait: (ctx: ChatFlowContext) => {
    let longPressTimer: NodeJS.Timeout | null = null;

    onButtonDoubleClick(null);

    display({
      status: "idle",
      emoji: "",
      RGB: "#000033",
      text: "Hold to answer...",
      rag_icon_visible: false,
    });

    onButtonPressed(() => {
      display({ text: "Listening...", RGB: "#00aa44" });
      longPressTimer = setTimeout(() => {
        longPressTimer = null;
        ctx.transitionTo("log_followup_listening");
      }, LONG_PRESS_MS);
    });

    onButtonReleased(() => {
      if (longPressTimer) {
        clearTimeout(longPressTimer);
        longPressTimer = null;
      }
      if (ctx.currentFlowName === "log_followup_wait") {
        display({ status: "idle", emoji: "", RGB: "#000033", text: "Hold to answer..." });
      }
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

    const { result, stop } = recordAudioManually(recordFilePath);

    onButtonReleased(() => {
      stop();
      // Pre-start FOLLOWUP_2 immediately on button release
      ctx.pendingLogResponseText = FOLLOWUP_2;
      ctx.logTTSPreStarted = true;
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
        saveLogEntry({ audioPath: recordFilePath, timestamp: Date.now(), type: "followup" });
        ctx.transitionTo("log_followup_response");
      })
      .catch((err) => {
        console.error("[log_followup_listening] Recording error:", err);
        ctx.transitionTo("sleep");
      });
  },
  log_followup_response: (ctx: ChatFlowContext) => {
    const fullText = ctx.logTTSPreStarted && ctx.pendingLogResponseText
      ? ctx.pendingLogResponseText
      : FOLLOWUP_2;

    display({ status: "answering...", emoji: "", RGB: "#00c8a3", text: fullText });

    onButtonPressed(() => {
      ctx.streamResponser.stop();
      ctx.transitionTo("sleep");
    });
    onButtonReleased(noop);

    if (!ctx.logTTSPreStarted) {
      void ctx.streamExternalReply(fullText);
    }
    ctx.logTTSPreStarted = false;
    ctx.pendingLogResponseText = "";

    ctx.streamResponser.getPlayEndPromise().then(() => {
      if (ctx.currentFlowName === "log_followup_response") {
        ctx.transitionTo("log_followup_2_wait");
      }
    });
  },
  log_followup_2_wait: (ctx: ChatFlowContext) => {
    let longPressTimer: NodeJS.Timeout | null = null;

    onButtonDoubleClick(null);

    display({
      status: "idle",
      emoji: "",
      RGB: "#000033",
      text: "Hold to answer...",
      rag_icon_visible: false,
    });

    onButtonPressed(() => {
      display({ text: "Listening...", RGB: "#00aa44" });
      longPressTimer = setTimeout(() => {
        longPressTimer = null;
        ctx.transitionTo("log_followup_2_listening");
      }, LONG_PRESS_MS);
    });

    onButtonReleased(() => {
      if (longPressTimer) {
        clearTimeout(longPressTimer);
        longPressTimer = null;
      }
      if (ctx.currentFlowName === "log_followup_2_wait") {
        display({ status: "idle", emoji: "", RGB: "#000033", text: "Hold to answer..." });
      }
    });

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
      stop();
      // Pre-start LOG_CONFIRMATION immediately
      ctx.pendingLogResponseText = LOG_CONFIRMATION;
      ctx.logTTSPreStarted = true;
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
        saveLogEntry({ audioPath: recordFilePath, timestamp: Date.now(), type: "followup" });
        ctx.transitionTo("log_confirmation");
      })
      .catch((err) => {
        console.error("[log_followup_2_listening] Recording error:", err);
        ctx.transitionTo("sleep");
      });
  },
  log_confirmation: (ctx: ChatFlowContext) => {
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
    ctx.pendingLogResponseText = "";

    ctx.streamResponser.getPlayEndPromise().then(() => {
      if (ctx.currentFlowName === "log_confirmation") {
        ctx.transitionTo("sleep");
      }
    });
  },
  eod_prompt: (ctx: ChatFlowContext) => {
    const eodText =
      "Before you wrap up today, anything you didn't log today that I could've helped with? And for anything you did log, what felt most urgent?";

    display({
      status: "answering...",
      emoji: "",
      RGB: "#ff9900",
      text: eodText,
    });

    onButtonPressed(() => {
      ctx.streamResponser.stop();
      ctx.transitionTo("sleep");
    });
    onButtonReleased(noop);

    void ctx.streamExternalReply(eodText);

    ctx.streamResponser.getPlayEndPromise().then(() => {
      if (ctx.currentFlowName === "eod_prompt") {
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
