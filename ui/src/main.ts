import "./style.css";
import { core } from "@tauri-apps/api";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  getCurrentWindow,
  currentMonitor,
  LogicalPosition,
  LogicalSize,
  Window,
} from "@tauri-apps/api/window";
import MarkdownIt from "markdown-it";
import markdownItKatex from "@vscode/markdown-it-katex";
import "katex/dist/katex.min.css";

// Configure markdown-it to use KaTeX
const md = new MarkdownIt();
md.use(markdownItKatex, { throwOnError: false });

// DOM Elements
const apiKeyInput = document.getElementById("api-key-input") as HTMLInputElement;
const geminiApiKeyInput = document.getElementById("gemini-api-key-input") as HTMLInputElement;
const settingsStatus = document.getElementById("settings-status") as HTMLParagraphElement;
const apiKeyStatusIcon = document.getElementById("api-key-status-icon") as HTMLSpanElement;
const settingsToggle = document.getElementById("settings-toggle") as HTMLButtonElement;
const settingsPanel = document.getElementById("settings-panel") as HTMLDivElement;
const clearChatButton = document.getElementById("clear-chat-button") as HTMLButtonElement;
const modelSelect = document.getElementById("model-select") as HTMLSelectElement;

const FADE_DURATION_SETTINGS = 80; // Duration for settings panel fade
// Define fixed window dimensions
const FIXED_WINDOW_WIDTH = 350;
const appWindow = getCurrentWindow();

// Create and add close button to settings panel
if (settingsPanel) {
  const closeButton = document.createElement("button");
  closeButton.id = "settings-close";
  closeButton.innerHTML = "×"; // Using × character for the X
  closeButton.title = "Close Settings";
  settingsPanel.appendChild(closeButton);

  // Add click handler for close button
  closeButton.addEventListener("click", () => {
    settingsPanel.classList.remove("fade-in-settings"); // Start fade-out
    setTimeout(() => {
      settingsPanel.style.display = "none";
    }, FADE_DURATION_SETTINGS);
    document.removeEventListener("click", handleClickOutsideSettings); // Remove listener
  });
}

const messageInput = document.getElementById("message-input") as HTMLTextAreaElement;
const inputImagePreview = document.getElementById("input-image-preview") as HTMLImageElement;
const chatHistory = document.getElementById("chat-history") as HTMLDivElement;
const ocrIconContainer = document.getElementById("ocr-icon-container") as HTMLDivElement;
const statusMessage = document.getElementById("status-message") as HTMLParagraphElement;

// Model mapping: Display Name -> OpenRouter Identifier
const modelMap: { [key: string]: string } = {
  "Deepseek R1 (free)": "deepseek/deepseek-r1:free",
  "Deepseek V3 (free)": "deepseek/deepseek-chat-v3-0324:free",
  "Gemini 2.0 Flash": "gemini-2.0-flash",
  "Gemini 2.5 Flash": "gemini-2.5-flash-preview-04-17",
  "Gemini 2.5 Flash (Thinking)": "gemini-2.5-flash-preview-04-17#thinking-enabled",
};

// Define the structure returned by the capture command
interface CaptureResult {
  ocr_text: string;
  image_base64: string | null;
  temp_path: string | null;
}

// Type for chat messages in the history
interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

// --- System Instruction (matches backend) ---
const SYSTEM_INSTRUCTION: string = `You provide accurate, factual answers
  - If you do not know the answer, make your best guess.`;

// --- Constants for History Management ---
const MAX_HISTORY_WORD_COUNT = 50000; // ADDED

// --- Helper function to count words ---
function getWordCount(text: string): number {
  // ADDED
  return text.split(/\s+/).filter(Boolean).length; // ADDED
}

// State Variables
let chatMessageHistory: ChatMessage[] = [];
let currentOcrText: string | null = null;
let currentImageBase64: string | null = null;
let currentTempScreenshotPath: string | null = null;
let currentAssistantMessageDiv: HTMLDivElement | null = null; // ADDED: To hold the div of the assistant's message being streamed
let currentAssistantContentDiv: HTMLDivElement | null = null; // ADDED: To hold the content part of the assistant's message
let isAIResponding: boolean = false; // ADDED: Flag to track if AI is currently responding

// --- Helper function to preprocess LaTeX delimiters ---
function preprocessLatex(content: string): string {
  // Replace \( ... \) with $ ... $
  content = content.replace(/\\\(/g, "$");
  content = content.replace(/\\\)/g, "$");
  // Replace \[ ... \] with $$ ... $$
  content = content.replace(/\\\[/g, "$$");
  content = content.replace(/\\\]/g, "$$");
  return content;
}

// --- Function to set initial window size and position ---
async function setInitialWindowGeometry() {
  try {
    const monitor = await currentMonitor();
    if (monitor) {
      const logicalMonitorHeight = monitor.size.height / monitor.scaleFactor;

      const targetLogicalY = logicalMonitorHeight - logicalMonitorHeight;

      await appWindow.setSize(new LogicalSize(FIXED_WINDOW_WIDTH, logicalMonitorHeight));
      await appWindow.setPosition(new LogicalPosition(0, targetLogicalY));
      console.log(`[WindowSetup] Window set to ${FIXED_WINDOW_WIDTH}x${logicalMonitorHeight} at (0, ${targetLogicalY})`);
    } else {
      // Fallback if monitor info isn't available (should be rare)
      await appWindow.setSize(new LogicalSize(FIXED_WINDOW_WIDTH, 750));
      console.warn("[WindowSetup] Could not get monitor info. Window with fixed size.");
    }
    await appWindow.setFocus();
  } catch (error) {
    console.error("[WindowSetup] Failed to set initial window geometry:", error);
  }
}

// --- Populate Model Select Dropdown ---
function populateModelSelect() {
  if (!modelSelect) return;
  modelSelect.innerHTML = ""; // Clear existing options
  for (const displayName in modelMap) {
    const option = document.createElement("option");
    option.value = modelMap[displayName]; // Use the identifier as the value
    option.textContent = displayName;
    modelSelect.appendChild(option);
  }
  console.log("Model select populated.");
}

// --- Load Initial State (includes API Key and Model) ---
async function loadInitialSettings() {
  // Load API Key
  try {
    const key = await invoke<string>("get_api_key");
    if (apiKeyInput) apiKeyInput.value = key || "";
    if (key) {
      console.log("API Key loaded.");
    } else {
      console.log("API Key not set.");
      // settingsStatus.textContent = 'API Key not set.'; // Optional: notify user
    }
  } catch (error) {
    console.error("Failed to load API key:", error);
    if (settingsStatus) settingsStatus.textContent = `Error loading key: ${error}`;
  }

  // Load Gemini API Key
  if (geminiApiKeyInput) {
    try {
      const geminiKey = await invoke<string>("get_gemini_api_key");
      geminiApiKeyInput.value = geminiKey || "";
      if (geminiKey) {
        console.log("Gemini API Key loaded.");
      } else {
        console.log("Gemini API Key not set.");
      }
    } catch (error) {
      console.error("Failed to load Gemini API key:", error);
      if (settingsStatus) settingsStatus.textContent = `Error loading Gemini key: ${error}`;
    }
  }

  // Populate model dropdown first
  populateModelSelect();

  // Load and set selected model
  if (modelSelect) {
    try {
      const selectedModelId = await invoke<string>("get_selected_model");
      console.log("Loaded selected model ID from backend:", selectedModelId);
      if (selectedModelId && modelSelect.options) {
        // Find the option with the matching value and select it
        let modelFound = false;
        for (let i = 0; i < modelSelect.options.length; i++) {
          if (modelSelect.options[i].value === selectedModelId) {
            modelSelect.selectedIndex = i;
            console.log("Set dropdown to:", selectedModelId);
            modelFound = true;
            break;
          }
        }
        if (!modelFound) {
          console.warn(
            `Saved model ID "${selectedModelId}" not found in dropdown options. Defaulting to first option.`,
          );
          // Optionally set to a default if the saved one is invalid or not found
          // modelSelect.selectedIndex = 0; // Or handle as needed
        }
      } else {
        console.log(
          "No selected model ID returned or modelSelect.options not available. Using default selection.",
        );
      }
    } catch (error) {
      console.error("Failed to load selected model:", error);
      if (settingsStatus) settingsStatus.textContent = `Error loading model: ${error}`;
    }
  }
}

// --- Helper to auto-resize textarea ---
const initialTextareaHeight = "calc(2em * 1.4)"; // Store initial height
function autoResizeTextarea() {
  if (!messageInput) return;
  // Temporarily shrink height to get accurate scrollHeight
  messageInput.style.height = "auto";
  // Set height based on content, but don't exceed a max
  const newHeight = Math.min(messageInput.scrollHeight, 200); // Limit max height to 200px (adjust as needed)
  messageInput.style.height = `${newHeight}px`;
  // Show scrollbar if content exceeds max height
  messageInput.style.overflowY = newHeight >= 200 ? "auto" : "hidden";
}

// --- Chat Functionality ---
async function addMessageToHistory(
  sender: "You" | "Shard",
  content: string,
  reasoning: string | null = null,
) {
  console.log(
    `addMessageToHistory called. Sender: ${sender}, Content: ${content}, Reasoning: ${reasoning}`,
  );
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message");
  messageDiv.classList.add(sender === "You" ? "user" : "assistant");

  const senderStrong = document.createElement("strong");
  senderStrong.textContent = sender;
  messageDiv.appendChild(senderStrong);

  // Add main content
  const contentDiv = document.createElement("div");
  contentDiv.classList.add("message-content");
  try {
    contentDiv.innerHTML = md.render(preprocessLatex(content)); // Render complete content with preprocessing
  } catch (e) {
    console.error("Error parsing markdown/katex:", e);
    contentDiv.textContent = content; // Fallback to text if parsing fails
  }
  messageDiv.appendChild(contentDiv);

  // If this is an assistant message being added (likely at the END of a stream or for non-streamed errors)
  // store its content div reference in case it was a non-streamed one (e.g. an error message directly added)
  if (sender === "Shard") {
    currentAssistantMessageDiv = messageDiv; // Store the whole message div
    currentAssistantContentDiv = contentDiv; // Store the content div specifically for updates
  }

  // ADDED: Check if this is the first USER message to show System Prompt
  if (sender === "You" && chatMessageHistory.length === 0) {
    const details = document.createElement("details");
    details.classList.add("reasoning-accordion"); // Reuse existing style
    details.style.marginTop = "10px"; // Add some space above the accordion

    const summary = document.createElement("summary");
    summary.textContent = "Show System Prompt";
    details.appendChild(summary);

    const promptContent = document.createElement("div");
    promptContent.classList.add("reasoning-content"); // Reuse existing style
    const pre = document.createElement("pre");
    pre.textContent = SYSTEM_INSTRUCTION;
    promptContent.appendChild(pre);

    details.appendChild(promptContent);
    messageDiv.appendChild(details); // Append to the user's message div
  }

  // Add to chatMessageHistory AFTER it's been decided what to display
  // For streamed messages, this will be updated in STREAM_END
  if (
    sender === "You" ||
    (sender === "Shard" && !chatHistory.querySelector(".message.assistant.thinking"))
  ) {
    // Add user messages immediately.
    // Add Shard messages only if it's not a streaming placeholder (which gets updated by STREAM_END)
    // This handles direct error messages from Shard added via addMessageToHistory.
    const role = sender === "You" ? "user" : "assistant";
    chatMessageHistory.push({ role, content });
  }

  // Add reasoning accordion if reasoning is present for assistant messages
  if (sender === "Shard" && reasoning) {
    const details = document.createElement("details");
    details.classList.add("reasoning-accordion");

    const summary = document.createElement("summary");
    summary.textContent = "Show Reasoning";
    details.appendChild(summary);

    const reasoningContent = document.createElement("div");
    reasoningContent.classList.add("reasoning-content");
    // Display reasoning as preformatted text for now
    const pre = document.createElement("pre");
    pre.textContent = reasoning;
    reasoningContent.appendChild(pre);

    details.appendChild(reasoningContent);
    messageDiv.appendChild(details);
  }

  // Prune history if it exceeds word count limit
  let currentTotalWords = chatMessageHistory.reduce(
    (sum, msg) => sum + getWordCount(msg.content),
    0,
  );
  while (currentTotalWords > MAX_HISTORY_WORD_COUNT && chatMessageHistory.length > 1) {
    const removedMessage = chatMessageHistory.shift();
    if (removedMessage) {
      currentTotalWords -= getWordCount(removedMessage.content);
    }
  }
  // console.log(`Current history word count: ${currentTotalWords}, messages: ${chatMessageHistory.length}`); // Optional: for debugging

  chatHistory.appendChild(messageDiv);
  chatHistory.scrollTop = chatHistory.scrollHeight; // Auto-scroll to bottom
}

// --- Helper to update Input Preview and Tooltip ---
function updateInputAreaForCapture() {
  if (currentOcrText) {
    messageInput.title = currentOcrText; // Set tooltip on input
  } else {
    messageInput.title = ""; // Clear tooltip
  }

  if (currentImageBase64) {
    inputImagePreview.src = `data:image/png;base64,${currentImageBase64}`;
    inputImagePreview.classList.remove("hidden");
  } else {
    inputImagePreview.src = "";
    inputImagePreview.classList.add("hidden");
  }
}

// --- Clear Chat Handler ---
async function clearChatHistory() {
  const bodyElement = document.body;

  // Start fade out
  console.log("Starting fade out for chat clear...");
  bodyElement.classList.add("fade-out");
  bodyElement.classList.remove("fade-in");

  // Wait for fade out to complete
  setTimeout(async () => {
  // Clear chat content while faded out
    if (chatHistory) chatHistory.innerHTML = "";
    console.log("Chat history cleared.");
    chatMessageHistory = [];

    // Hide clear button again
    if (clearChatButton) clearChatButton.classList.add("hidden");

    // Clear capture state and cleanup temp file if necessary
    if (currentTempScreenshotPath) {
      console.log("Cleanup requested for temp screenshot:", currentTempScreenshotPath);
      invoke("cleanup_temp_screenshot", { path: currentTempScreenshotPath })
        .then(() => console.log("Temp screenshot cleanup successful."))
        .catch((err) => console.error("Error cleaning up temp screenshot:", err));
    }
    currentOcrText = null;
    currentImageBase64 = null;
    currentTempScreenshotPath = null;
    updateInputAreaForCapture(); // Clear preview and tooltip

    if (statusMessage) statusMessage.textContent = ""; // Clear status

    // Start fade in after everything is resized
    console.log("Starting fade in after chat clear...");
    bodyElement.classList.remove("fade-out");
    bodyElement.classList.add("fade-in");

    // Clean up fade-in class after animation completes
    setTimeout(() => {
      bodyElement.classList.remove("fade-in");
    }, FADE_DURATION);

  }, FADE_DURATION);
}

// --- Capture OCR Handler ---
async function handleCaptureOcr() {
  console.log("Capture OCR initiated");
  if (statusMessage) {
    statusMessage.textContent = "Starting screen capture...";
    statusMessage.style.display = "block";
  }
  // Clear previous capture state *before* starting new capture
  if (currentTempScreenshotPath) {
    console.log("Cleaning up previous temp screenshot:", currentTempScreenshotPath);
    await invoke("cleanup_temp_screenshot", { path: currentTempScreenshotPath }).catch((err) =>
      console.error("Error cleaning up previous temp screenshot:", err),
    );
    currentTempScreenshotPath = null; // Ensure path is cleared even if cleanup fails
  }
  currentOcrText = null;
  currentImageBase64 = null;
  updateInputAreaForCapture();

  // Visually indicate loading
  if (ocrIconContainer) ocrIconContainer.style.opacity = "0.5";

  try {
    const result = await core.invoke<CaptureResult>("capture_interactive_and_ocr");
    console.log("Capture Result:", result);

    currentOcrText = result.ocr_text;
    currentImageBase64 = result.image_base64;
    currentTempScreenshotPath = result.temp_path;

    updateInputAreaForCapture(); // Update input tooltip and image preview

    if (statusMessage) {
      if (currentOcrText || currentImageBase64) {
        statusMessage.textContent = "Capture complete. OCR text added as tooltip to input.";
        // Auto-hide status after a delay
        setTimeout(() => {
          if (
            statusMessage &&
            statusMessage.textContent === "Capture complete. OCR text added as tooltip to input."
          ) {
            statusMessage.style.display = "none";
            statusMessage.textContent = "";
          }
        }, 4000);
      } else {
        statusMessage.textContent = "Capture complete, but no image or text was processed.";
      }
    }
  } catch (error) {
    console.error("Error during interactive capture/OCR:", error);
    const errorMessage = typeof error === "string" ? error : "Capture cancelled or failed.";
    // Clear any potentially partially set state
    currentOcrText = null;
    currentImageBase64 = null;
    currentTempScreenshotPath = null; // Ensure path isn't left hanging on error
    updateInputAreaForCapture();

    if (statusMessage) {
      statusMessage.textContent = `Error: ${errorMessage}`;
      // Auto-hide error after a delay
      setTimeout(() => {
        if (statusMessage && statusMessage.textContent === `Error: ${errorMessage}`) {
          statusMessage.style.display = "none";
          statusMessage.textContent = "";
        }
      }, 5000);
    }
  } finally {
    // Re-enable icon
    if (ocrIconContainer) ocrIconContainer.style.opacity = "1";
  }
}

// --- Send Message Handler ---
async function handleSendMessage() {
  if (isAIResponding) {
    console.log("handleSendMessage: AI is currently responding. New message blocked.");
    return; // Prevent sending a new message while AI is responding
  }

  let userTypedText = messageInput.value.trim();
  let textToSend = userTypedText;
  let textToDisplay = userTypedText;
  let tempPathToClean: string | null = null; // Hold path for cleanup *after* sending

  // Check if there's captured OCR text to prepend
  if (currentOcrText) {
    const formattedOcr = `\n OCR Text: ${currentOcrText}`;
    textToSend = userTypedText ? `${userTypedText}\n\n${formattedOcr}` : formattedOcr;
    textToDisplay = textToSend; // Display the combined text

    // Prepare state to be cleared AFTER successful send
    tempPathToClean = currentTempScreenshotPath;
    currentOcrText = null;
    currentImageBase64 = null;
    currentTempScreenshotPath = null;
    // updateInputAreaForCapture(); // Clear preview/tooltip *after* send succeeds or fails
  } else if (!userTypedText) {
    console.log("handleSendMessage: No text typed and no captured OCR text.");
    return; // Nothing to send
  }

  // Add user's current message to history right before sending
  // This was previously done in addMessageToHistory, but to ensure the API call
  // gets the absolute latest state including the current message, we adjust.
  // However, addMessageToHistory ALREADY adds the user message to the visual chat
  // and to chatMessageHistory. So, the history is up-to-date.

  addMessageToHistory("You", textToDisplay); // This will also add it to chatMessageHistory

  // Optimistically resize window to maximum height expecting a potentially long response

  // Prepare messages for the backend
  // The 'textToDisplay' is the most recent user message.
  // chatMessageHistory already contains all prior messages, including the one just added by addMessageToHistory.
  const messagesToSendToBackend = [...chatMessageHistory]; // MODIFIED: Use the updated chatMessageHistory

  messageInput.value = ""; // Clear input field now
  // messageInput.disabled = true; // MODIFIED: Allow typing while AI responds
  if (messageInput.title) messageInput.title = ""; // Clear tooltip immediately
  messageInput.style.height = initialTextareaHeight; // Reset height
  messageInput.style.overflowY = "hidden"; // Hide scrollbar again
  if (!inputImagePreview.classList.contains("hidden")) {
    inputImagePreview.classList.add("hidden"); // Hide preview immediately
    inputImagePreview.src = "";
  }

  // Show clear button if it was hidden
  if (clearChatButton?.classList.contains("hidden")) {
    clearChatButton.classList.remove("hidden");
  }

  // Show thinking indicator / initial placeholder for Shard's response
  const assistantMessagePlaceholder = document.createElement("div");
  assistantMessagePlaceholder.classList.add("message", "assistant", "streaming"); // New class for styling streamed message
  const senderStrong = document.createElement("strong");
  senderStrong.textContent = "Shard";
  assistantMessagePlaceholder.appendChild(senderStrong);

  currentAssistantContentDiv = document.createElement("div"); // Create the content div
  currentAssistantContentDiv.classList.add("message-content");
  // Use the new getStreamingDots() function for the initial indicator
  currentAssistantContentDiv.innerHTML = ""; // Clear any default content
  currentAssistantContentDiv.appendChild(getStreamingDots());
  assistantMessagePlaceholder.appendChild(currentAssistantContentDiv);

  currentAssistantMessageDiv = assistantMessagePlaceholder; // Store reference to the whole message

  if (chatHistory) {
    chatHistory.appendChild(assistantMessagePlaceholder);
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  isAIResponding = true; // Set flag as AI is about to respond
  try {
    // Invoke send_text_to_model. It no longer directly returns the message content.
    await core.invoke("send_text_to_model", {
      messages: messagesToSendToBackend,
      window: Window.getCurrent(),
    }); // Pass the current window
    console.log("send_text_to_model invoked. Waiting for stream events.");

    // The rest of the logic (removeThinkingIndicator, addMessageToHistory for Shard)
    // will now be handled by the STREAM_CHUNK, STREAM_END, and STREAM_ERROR event listeners.

    // Cleanup temp file ONLY after successful command invocation (actual success is via stream)
    if (tempPathToClean) {
      console.log(
        "Cleaning up temp screenshot after invoking send_text_to_model:",
        tempPathToClean,
      );
      invoke("cleanup_temp_screenshot", { path: tempPathToClean }).catch((err) =>
        console.error("Error cleaning up temp screenshot post-invoke:", err),
      );
    }
  } catch (error) {
    // This catch block handles errors from the invoke call itself (e.g., backend not reachable)
    // Errors from the model generation will be handled by STREAM_ERROR listener.
    console.error("Failed to invoke send_text_to_model:", error);
    if (currentAssistantContentDiv) {
      currentAssistantContentDiv.innerHTML = md.render(
        preprocessLatex(`Error invoking model: ${error}`),
      );
    } else {
      // If even the placeholder wasn't created, add a new error message
      addMessageToHistory("Shard", `Error invoking model: ${error}`);
    }
    if (currentAssistantMessageDiv) {
      currentAssistantMessageDiv.classList.remove("streaming"); // Remove streaming class if error occurs here
      currentAssistantMessageDiv.classList.add("error"); // Optional: add error class
    }
    isAIResponding = false; // Reset flag on invoke error

    // Even on error, if we *tried* to send OCR text, attempt cleanup
    if (tempPathToClean) {
      console.warn("Cleaning up temp screenshot after failed invoke:", tempPathToClean);
      invoke("cleanup_temp_screenshot", { path: tempPathToClean }).catch((err) =>
        console.error("Error cleaning up temp screenshot post-failure:", err),
      );
    }
  } finally {
    // Re-enable input regardless of success/failure OF THE INVOKE CALL
    // Actual message completion enables it in STREAM_END or STREAM_ERROR
    // if (messageInput) { // MODIFIED: No longer disabling/enabling here for this purpose
    //   messageInput.disabled = false;
    //   // messageInput.focus();
    // }
    // Ensure preview/tooltip are cleared in case they weren't before
    updateInputAreaForCapture();
  }
}

// --- Event Listeners ---

// Define interfaces for stream payloads
interface StreamChunkPayload {
  delta?: string | null;
}
interface StreamEndPayload {
  full_content: string;
  reasoning?: string | null;
}
interface StreamErrorPayload {
  error: string;
}

let unlistenStreamChunk: (() => void) | null = null;
let unlistenStreamEnd: (() => void) | null = null;
let unlistenStreamError: (() => void) | null = null;

// Buffer and flag for batched animation of stream chunks
let streamDeltaBuffer = ""; // ADDED: Accumulates deltas
let streamAnimationFrameRequested = false; // ADDED: Tracks if an animation frame is pending

// ADDED: Configuration for sub-chunking large pieces of text
const MAX_SUB_CHUNK_LENGTH = 70; // Characters per animated sub-chunk
const SUB_CHUNK_ANIMATION_DELAY = 50; // Milliseconds delay between animating sub-chunks

// --- Helper function to create streaming dots ---
function getStreamingDots(): HTMLSpanElement {
  const dotsContainer = document.createElement("span");
  dotsContainer.classList.add("streaming-dots");
  for (let i = 0; i < 3; i++) {
    const dot = document.createElement("span");
    dotsContainer.appendChild(dot);
  }
  return dotsContainer;
}

async function setupStreamListeners() {
  if (unlistenStreamChunk) unlistenStreamChunk();
  unlistenStreamChunk = await listen<StreamChunkPayload>("STREAM_CHUNK", (event) => {
    if (event.payload.delta) {
      streamDeltaBuffer += event.payload.delta;
    }

    if (!streamAnimationFrameRequested && currentAssistantContentDiv) {
      streamAnimationFrameRequested = true;
      requestAnimationFrame(() => {
        if (!currentAssistantContentDiv) {
          // Double check in case it became null
          streamAnimationFrameRequested = false;
          streamDeltaBuffer = ""; // Clear buffer if no target
          return;
        }

        const currentBatchText = streamDeltaBuffer;
        streamDeltaBuffer = ""; // Clear buffer for next frame's network chunks
        streamAnimationFrameRequested = false; // Reset flag for next frame

        if (currentBatchText) {
          // The initial display is now also .streaming-dots, so this specific check for "dots-container" is less critical
          // but the general logic of removing dots before adding text is sound.
          if (currentAssistantContentDiv.innerHTML.includes("dots-container")) { // This will likely be false now
            currentAssistantContentDiv.innerHTML = ""; // Clear initial thinking dots (if they were the old style)
          }

          // Remove any existing streaming dots before adding new text
          const existingDots = currentAssistantContentDiv.querySelector(".streaming-dots");
          if (existingDots) {
            existingDots.remove();
          }

          // Function to animate text piece by piece
          function animateTextSequentially(textToProcess: string) {
            if (!textToProcess || !currentAssistantContentDiv) return;

            const subChunk = textToProcess.substring(0, MAX_SUB_CHUNK_LENGTH);
            const remainingText = textToProcess.substring(MAX_SUB_CHUNK_LENGTH);

            const newSpan = document.createElement("span");
            newSpan.innerHTML = md.renderInline(preprocessLatex(subChunk)); // Render this piece with preprocessing
            newSpan.style.opacity = "0";
            newSpan.style.transition = "opacity 0.3s ease-out";
            currentAssistantContentDiv.appendChild(newSpan);

            requestAnimationFrame(() => {
              // Fade in this piece
              newSpan.style.opacity = "1";
            });

            if (chatHistory) {
              chatHistory.scrollTop = chatHistory.scrollHeight;
            }

            if (remainingText) {
              setTimeout(() => {
                animateTextSequentially(remainingText);
              }, SUB_CHUNK_ANIMATION_DELAY);
            } else {
              // Append streaming dots after the last sub-chunk is animated
              if (currentAssistantContentDiv) {
                currentAssistantContentDiv.appendChild(getStreamingDots());
                if (chatHistory) chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll again after adding dots
              }
            }
          }
          animateTextSequentially(currentBatchText); // Start processing the batch
        } else if (currentAssistantContentDiv.innerHTML !== "" && !currentAssistantContentDiv.querySelector(".streaming-dots")) {
          // If buffer was empty but there's content and no dots, add dots (e.g. after clearing initial dots)
          currentAssistantContentDiv.appendChild(getStreamingDots());
          if (chatHistory) chatHistory.scrollTop = chatHistory.scrollHeight;
        }
      });
    } else if (!currentAssistantContentDiv && streamDeltaBuffer) {
      console.warn(
        "STREAM_CHUNK: currentAssistantContentDiv is null, but deltaBuffer has content:",
        streamDeltaBuffer,
      );
      streamDeltaBuffer = "";
      streamAnimationFrameRequested = false;
    }
  });

  if (unlistenStreamEnd) unlistenStreamEnd();
  unlistenStreamEnd = await listen<StreamEndPayload>("STREAM_END", async (event) => {
    console.log("STREAM_END received:", event.payload);
    if (currentAssistantMessageDiv && currentAssistantContentDiv) {
      // Remove streaming dots before setting final content
      const existingDots = currentAssistantContentDiv.querySelector(".streaming-dots");
      if (existingDots) {
        existingDots.remove();
      }
      currentAssistantContentDiv.innerHTML = md.render(preprocessLatex(event.payload.full_content)); // Final render with preprocessing
      currentAssistantMessageDiv.classList.remove("streaming");

      // Add reasoning if present
      if (event.payload.reasoning) {
        const details = document.createElement("details");
        details.classList.add("reasoning-accordion");
        const summary = document.createElement("summary");
        summary.textContent = "Show Reasoning";
        details.appendChild(summary);
        const reasoningContentEl = document.createElement("div");
        reasoningContentEl.classList.add("reasoning-content");
        const pre = document.createElement("pre");
        pre.textContent = event.payload.reasoning;
        reasoningContentEl.appendChild(pre);
        details.appendChild(reasoningContentEl);
        currentAssistantMessageDiv.appendChild(details);
      }

      // Update chatMessageHistory with the complete message
      const existingEntryIndex = chatMessageHistory.findIndex(
        (msg) => msg.role === "assistant" && msg.content === "Thinking...",
      ); // Placeholder text if used
      if (existingEntryIndex > -1) {
        chatMessageHistory[existingEntryIndex].content = event.payload.full_content;
      } else {
        // If no placeholder was there (e.g. direct error), or to be safe, push a new one.
        // However, the placeholder should be handled by the currentAssistantMessageDiv logic.
        // The main purpose here is to ensure the history array has the final content.
        // Let's refine the logic for adding to chatMessageHistory:
        // The 'addMessageToHistory' function itself is not ideal for streaming placeholders.
        // We should add to chatMessageHistory here, at the END.
        chatMessageHistory.push({ role: "assistant", content: event.payload.full_content });
      }
    }
    if (messageInput) {
      messageInput.disabled = false; // This now means "allow sending again"
      // messageInput.focus();
    }
    currentAssistantMessageDiv = null;
    currentAssistantContentDiv = null;
    isAIResponding = false; // Reset flag as AI has finished responding
    if (chatHistory) chatHistory.scrollTop = chatHistory.scrollHeight;
  });

  if (unlistenStreamError) unlistenStreamError();
  unlistenStreamError = await listen<StreamErrorPayload>("STREAM_ERROR", (event) => {
    console.error("STREAM_ERROR received:", event.payload);
    if (currentAssistantMessageDiv && currentAssistantContentDiv) {
      // Remove streaming dots before setting error content
      const existingDots = currentAssistantContentDiv.querySelector(".streaming-dots");
      if (existingDots) {
        existingDots.remove();
      }
      currentAssistantContentDiv.innerHTML = md.render(
        preprocessLatex(`Error: ${event.payload.error}`),
      ); // Preprocess LaTeX
      currentAssistantMessageDiv.classList.remove("streaming");
      currentAssistantMessageDiv.classList.add("error"); // Optional: add error class for styling
    } else {
      // If no placeholder, add a new message for the error
      addMessageToHistory("Shard", `Error: ${event.payload.error}`);
    }
    if (messageInput) {
      messageInput.disabled = false; // This now means "allow sending again"
      // messageInput.focus();
    }
    currentAssistantMessageDiv = null;
    currentAssistantContentDiv = null;
    isAIResponding = false; // Reset flag on stream error
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  loadInitialSettings();
  setupStreamListeners(); // ADDED: Setup listeners on DOM load
  await setInitialWindowGeometry(); // Set fixed window size and position

  // --- Click-Through Logic ---
  // Ensure the window is interactive when it gains focus
  appWindow.onFocusChanged(async ({ payload: focused }) => {
    console.log(`[ClickThrough] onFocusChanged event. Focused: ${focused}`); // Log focus change
    if (focused) {
      try {
        await appWindow.setIgnoreCursorEvents(false);
        console.log("[ClickThrough] Window focused, cursor events enabled.");
      } catch (error) {
        console.error("[ClickThrough] Error enabling cursor events on focus:", error);
      }
    } else {
      // Optional: Log when window loses focus, might be relevant
      console.log("[ClickThrough] Window lost focus.");
    }
  });

  // Handle clicks to potentially enable click-through
  document.addEventListener('mousedown', async (event) => {
    const target = event.target as HTMLElement;
    // Basic check to avoid errors if target is not an HTMLElement (e.g., SVGElement in some cases, though less common for this specific problem)
    if (!target || typeof target.closest !== 'function') {
      console.log("[ClickThrough] Event target is not an HTMLElement or doesn't support 'closest'. Ignoring.");
      return;
    }
    console.log(`[ClickThrough] Mousedown event. Target: <${target.tagName}> id='${target.id || "none"}' class='${target.className || "none"}'`);

    // Define selectors for all elements that should remain interactive
    const interactiveSelectors = [
      '#message-input', '#input-image-preview',
      '#ocr-icon-container', '#clear-chat-button',
      '#settings-toggle', '#settings-panel', // settings-panel and all its children
      '#chat-history > *', // Any direct child of chat-history (messages, accordions, etc.)
      '#input-area',
      // General HTML tags that are usually interactive by nature
      'button', 'textarea', 'input', 'select', 'details', 'summary',
      // Potentially add specific IDs/classes of scrollbars if they become an issue.
      // Add any other specific interactive elements by ID or class if needed
    ];

    let isInteractiveClick = false;
    let matchedSelector = "none";
    for (const selector of interactiveSelectors) {
      if (target.closest(selector)) {
        isInteractiveClick = true;
        matchedSelector = selector;
        break;
      }
    }

    if (isInteractiveClick) {
      console.log(`[ClickThrough] Click target matched interactive selector: '${matchedSelector}'. Window remains interactive.`);
      // Ensure the window is interactive if an interactive element is clicked.
      try {
        await appWindow.setIgnoreCursorEvents(false);
        // console.log("[ClickThrough] Ensured cursor events are enabled due to interactive click.");
      } catch (error) {
        console.error("[ClickThrough] Error ensuring cursor events enabled on interactive click:", error);
      }
    } else {
      console.log("[ClickLogic] Click target did not match interactive selectors. Emitting 'js-request-toggle-window' to backend.");
      try {
        // We'll need a simple Rust command to re-emit an event from Rust.
        // Let's assume a command like `trigger_toggle_window_event` for now.
        // This command will live in Rust and do app_handle.emit("toggle-main-window", ())
        await invoke("trigger_backend_window_toggle");
        console.log("[ClickLogic] Successfully requested backend window toggle.");
      } catch (error) {
        console.error("[ClickLogic] Error requesting backend window toggle:", error);
      }
    }
  });
  // --- End Click-Through Logic ---

  if (apiKeyInput) {
    // Add input event listener with debounce for auto-saving
    let saveTimeout: number;
    apiKeyInput.addEventListener("input", () => {
      clearTimeout(saveTimeout);
      const key = apiKeyInput.value.trim();

      // Update immediately if clearing the key
      if (!key) {
        core.invoke("set_api_key", { key });
        if (settingsStatus) settingsStatus.textContent = "API Key cleared";
        if (apiKeyStatusIcon) apiKeyStatusIcon.classList.remove("visible");
        setTimeout(() => {
          if (settingsStatus && settingsStatus.textContent === "API Key cleared")
            settingsStatus.textContent = "";
        }, 3000);
        return;
      }

      // Otherwise, debounce the save
      saveTimeout = setTimeout(async () => {
        try {
          await core.invoke("set_api_key", { key });
          if (settingsStatus) settingsStatus.textContent = "";
          if (apiKeyStatusIcon) {
            apiKeyStatusIcon.classList.add("visible");
            setTimeout(() => {
              apiKeyStatusIcon.classList.remove("visible");
            }, 2000);
          }
        } catch (error) {
          console.error("Failed to save API key:", error);
          if (apiKeyStatusIcon) apiKeyStatusIcon.classList.remove("visible");
          if (settingsStatus) settingsStatus.textContent = "Error saving API key.";
        }
      }, 500); // Wait 500ms after last keystroke before saving
    });
  }

  // Listener for Gemini API Key input
  if (geminiApiKeyInput) {
    let geminiSaveTimeout: number;
    geminiApiKeyInput.addEventListener("input", () => {
      clearTimeout(geminiSaveTimeout);
      const key = geminiApiKeyInput.value.trim();

      if (!key) {
        core.invoke("set_gemini_api_key", { key });
        if (settingsStatus) settingsStatus.textContent = "Gemini API Key cleared";
        if (apiKeyStatusIcon) apiKeyStatusIcon.classList.remove("visible");
        setTimeout(() => {
          if (settingsStatus && settingsStatus.textContent === "Gemini API Key cleared")
            settingsStatus.textContent = "";
        }, 3000);
        return;
      }

      geminiSaveTimeout = setTimeout(async () => {
        try {
          await core.invoke("set_gemini_api_key", { key });
          if (settingsStatus) settingsStatus.textContent = "";
          if (apiKeyStatusIcon) {
            apiKeyStatusIcon.classList.add("visible");
            setTimeout(() => {
              apiKeyStatusIcon.classList.remove("visible");
            }, 2000);
          }
        } catch (error) {
          console.error("Failed to save Gemini API key:", error);
          if (apiKeyStatusIcon) apiKeyStatusIcon.classList.remove("visible");
          if (settingsStatus) settingsStatus.textContent = "Error saving Gemini API key.";
        }
      }, 500);
    });
  }

  if (messageInput) {
    // Set initial height correctly
    messageInput.style.height = initialTextareaHeight;
    messageInput.style.overflowY = "hidden";

    // Add input event listener for auto-resizing
    messageInput.addEventListener("input", autoResizeTextarea);

    messageInput.addEventListener("keypress", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !isAIResponding) {
        // Allow Shift+Enter for newlines if desired
        event.preventDefault(); // Prevent default Enter behavior (like adding newline)
        handleSendMessage();
      }
    });
  }

  if (settingsToggle) {
    settingsToggle.addEventListener("click", (_event) => {
      if (settingsPanel.style.display === "none" || settingsPanel.style.display === "") {
        settingsPanel.style.display = "block";
        // Force reflow before adding class to ensure transition happens
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            settingsPanel.classList.add("fade-in-settings");
          });
        });
        setTimeout(() => document.addEventListener("click", handleClickOutsideSettings), 0);
      } else {
        settingsPanel.classList.remove("fade-in-settings"); // Start fade-out
        setTimeout(() => {
          settingsPanel.style.display = "none";
        }, FADE_DURATION_SETTINGS);
        document.removeEventListener("click", handleClickOutsideSettings);
      }
    });
  }

  if (clearChatButton) {
    clearChatButton.addEventListener("click", clearChatHistory);
  }

  // Model selection change listener
  if (modelSelect) {
    modelSelect.addEventListener("change", async (event) => {
      const selectedModelId = (event.target as HTMLSelectElement).value;
      console.log("Model selection changed to:", selectedModelId);
      try {
        await invoke("set_selected_model", { modelName: selectedModelId });
        console.log("Successfully saved selected model:", selectedModelId);
      } catch (error) {
        console.error("Failed to save selected model:", error);
        if (settingsStatus) settingsStatus.textContent = `Error saving model: ${error}`;
      }
    });
  }

  // Add listener to the OCR icon container
  if (ocrIconContainer) {
    ocrIconContainer.addEventListener("click", handleCaptureOcr);
  }
});

// --- Event Listener for Window Toggle ---
const FADE_DURATION = 200; // ms - Should match CSS transition duration

// --- Function to handle clicks outside the settings panel ---
function handleClickOutsideSettings(event: MouseEvent) {
  if (settingsPanel && settingsPanel.style.display === "block") {
    // Check if the click is outside the panel AND not on the toggle button itself
    if (!settingsPanel.contains(event.target as Node) && event.target !== settingsToggle) {
      settingsPanel.classList.remove("fade-in-settings"); // Start fade-out
      setTimeout(() => {
        settingsPanel.style.display = "none";
      }, FADE_DURATION_SETTINGS);
      document.removeEventListener("click", handleClickOutsideSettings);
    }
  }
}
listen("toggle-main-window", async () => {
  console.log("toggle-main-window event received from backend!");

  // Ensure cursor events are enabled when the window is toggled
  try {
    await appWindow.setIgnoreCursorEvents(false);
    console.log("[ToggleWindow] Ensured cursor events are enabled.");
  } catch (error) {
    console.error("[ToggleWindow] Error enabling cursor events on toggle:", error);
  }

  const mainWindow = await Window.getByLabel("main");
  if (!mainWindow) {
    console.error("Main window not found by label 'main'!");
    return;
  }

  const isVisible = await mainWindow.isVisible();
  const bodyElement = document.body;

  if (isVisible) {
    console.log("Fading out window...");
    bodyElement.classList.add("fade-out");
    bodyElement.classList.remove("fade-in"); // Ensure fade-in is removed

    // Wait for fade-out animation to complete before hiding
    setTimeout(async () => {
      await mainWindow.hide();
      bodyElement.classList.remove("fade-out"); // Clean up class
      console.log("Window hidden.");
    }, FADE_DURATION);
  } else {
    console.log("Fading in window...");
    // Ensure opacity is 0 before showing if using fade-in class
    bodyElement.style.opacity = "0"; // Start transparent
    bodyElement.classList.remove("fade-out"); // Ensure fade-out is removed

    await mainWindow.show(); // Show the (transparent) window
    if (messageInput) {
      messageInput.focus(); // Focus the input field when window is shown
    }

    // Force reflow/repaint before adding fade-in class might be needed in some cases
    requestAnimationFrame(() => {
      bodyElement.style.opacity = ""; // Reset opacity for CSS transition
      bodyElement.classList.add("fade-in");
      console.log("Fade-in class added.");

      // Optional: Remove fade-in class after animation completes to reset state
      setTimeout(() => {
        bodyElement.classList.remove("fade-in");
      }, FADE_DURATION);
    });
  }
});

console.log("Frontend listener for toggle-main-window set up.");

