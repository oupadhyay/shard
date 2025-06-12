import "./style.css";
import DOMPurify from "dompurify";
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

// Safe rendering helper with DOMPurify sanitization
function safeRender(content: string): string {
  const rawHtml = md.render(preprocessLatex(content));
  return DOMPurify.sanitize(rawHtml);
}

function safeRenderInline(content: string): string {
  const rawHtml = md.renderInline(preprocessLatex(content));
  return DOMPurify.sanitize(rawHtml);
}

md.use(markdownItKatex, { throwOnError: false });

// DOM Elements
const apiKeyInput = document.getElementById("api-key-input") as HTMLInputElement;
const geminiApiKeyInput = document.getElementById("gemini-api-key-input") as HTMLInputElement;
const settingsStatus = document.getElementById("settings-status") as HTMLParagraphElement;
const apiKeyStatusIcon = document.getElementById("api-key-status-icon") as HTMLSpanElement;
const settingsToggle = document.getElementById("settings-toggle") as HTMLButtonElement;
const settingsPanel = document.getElementById("settings-panel") as HTMLDivElement;
const clearChatButton = document.getElementById("clear-chat-button") as HTMLButtonElement;
const clearIcon = document.getElementById("clear-icon") as SVGElement | null;
const undoIcon = document.getElementById("undo-icon") as SVGElement | null;
const modelSelect = document.getElementById("model-select") as HTMLSelectElement;
const webSearchToggle = document.getElementById("web-search-toggle") as HTMLInputElement;

const FADE_DURATION_SETTINGS = 80; // Duration for settings panel fade
// Define fixed window dimensions
const FIXED_WINDOW_WIDTH = 300;
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

const messageInput = document.getElementById("message-input") as HTMLDivElement; // MODIFIED: HTMLDivElement
const chatHistory = document.getElementById("chat-history") as HTMLDivElement;
const ocrIconContainer = document.getElementById("ocr-icon-container") as HTMLDivElement;
const statusMessage = document.getElementById("status-message") as HTMLParagraphElement;

// Model mapping: Display Name -> OpenRouter Identifier
const modelMap: { [key: string]: string } = {
  // "Deepseek R1 (free)": "deepseek/deepseek-r1:free",
  "Deepseek R1 (05-28)": "deepseek/deepseek-r1-0528:free",
  "Deepseek V3 (03-24)": "deepseek/deepseek-chat-v3-0324:free",
  "Gemini 2.0 Flash": "gemini-2.0-flash",
  "Gemini 2.5 Flash (05-20)": "gemini-2.5-flash-preview-05-20",
  "Gemini 2.5 Flash (Thinking)": "gemini-2.5-flash-preview-05-20#thinking-enabled",
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
  image_base64_data?: string | null;
  image_mime_type?: string | null;
}

// --- System Instruction (matches backend) ---
const SYSTEM_INSTRUCTION: string = `You are a helpful assistant that provides accurate, factual answers. If you do not know the answer, make your best guess. You are casual in tone and prefer concise responses. Avoid starting responses with \"**\". You prefer bulleted lists when needed but never use nested lists/sub-bullets. Use markdown for code blocks and links. For math: use $$....$$ for display equations (full-line) and \\(...\\) for inline math. Never mix $ and $$ syntax.`;

// --- Constants for History Management ---
const MAX_HISTORY_WORD_COUNT = 50000; // ADDED

// --- Helper function to count words ---
function getWordCount(text: string): number {
  // ADDED
  return text.split(/\s+/).filter(Boolean).length; // ADDED
}

// Backup interface for undo functionality
interface ChatBackup {
  chatMessageHistory: ChatMessage[];
  chatHistoryHTML: string;
  currentOcrText: string | null;
  currentImageBase64: string | null;
  currentImageMimeType: string | null;
  currentTempScreenshotPath: string | null;
}

// State Variables
let chatMessageHistory: ChatMessage[] = [];
let currentOcrText: string | null = null;
let currentImageBase64: string | null = null;
let currentImageMimeType: string | null = null; // ADDED
let currentTempScreenshotPath: string | null = null;
let currentAssistantMessageDiv: HTMLDivElement | null = null; // ADDED: To hold the div of the assistant's message being streamed
let currentAssistantContentDiv: HTMLDivElement | null = null; // ADDED: To hold the content part of the assistant's message
let isAIResponding: boolean = false; // ADDED: Flag to track if AI is currently responding
let responseCounter: number = 0; // ADDED: Simple counter to prevent stream cross-talk

// Undo functionality state
let chatBackup: ChatBackup | null = null;
let canUndo: boolean = false;

// Map to track message divs per response counter
const responseDivMap = new Map<
  number,
  { messageDiv: HTMLDivElement; contentDiv: HTMLDivElement }
>();

// --- Helper function to preprocess LaTeX delimiters ---
function preprocessLatex(content: string): string {
  // Replace \( ... \) with $ ... $
  content = content.replace(/\\\(/g, "$");
  content = content.replace(/\\\)/g, "$");
  // Replace \[ ... \] with $$ ... $$
  content = content.replace(/\\\[/g, "$$");
  content = content.replace(/\\\]/g, "$$");

  // Remove newlines within $$ ... $$ blocks, replacing them with a space
  content = content.replace(/\$\$(.*?)\$\$/gs, (_, innerContent) => {
    // Replace one or more newline characters (Unix, Windows, old Mac) with a single space
    const cleanedInnerContent = innerContent.replace(/[\n\r]+/g, " ");
    return `$$${cleanedInnerContent}$$`;
  });

  return content;
}

// --- Helper function to clean LaTeX/MathML markup from text ---
function cleanLatexMarkup(content: string): string {
  // Remove LaTeX displaystyle blocks like {\displaystyle ...}
  content = content.replace(/\{\s*\\displaystyle\s+[^}]*\}/g, "");

  // Remove MathML-style markup with nested tags
  content = content.replace(/<[^>]*>/g, "");

  // Remove excessive whitespace between mathematical symbols
  content = content.replace(/\s+/g, " ");

  // Clean up common LaTeX commands and keep just the symbols
  content = content.replace(/\\mathbf\s*\{\s*([^}]+)\s*\}/g, "$1");
  content = content.replace(/\\varepsilon/g, "ε");
  content = content.replace(/\\mu/g, "μ");
  content = content.replace(/\\nabla/g, "∇");
  content = content.replace(/\\cdot/g, "⋅");
  content = content.replace(/\\times/g, "×");
  content = content.replace(/\\partial/g, "∂");
  content = content.replace(/\\frac\s*\{\s*([^}]+)\s*\}\s*\{\s*([^}]+)\s*\}/g, "($1)/($2)");

  // Remove remaining LaTeX markup patterns
  content = content.replace(/\\\w+\s*\{[^}]*\}/g, "");
  content = content.replace(/\\\w+/g, "");
  content = content.replace(/[\{\}]/g, "");

  // Clean up extra spaces and normalize
  content = content.replace(/\s+/g, " ").trim();

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
      console.log(
        `[WindowSetup] Window set to ${FIXED_WINDOW_WIDTH}x${logicalMonitorHeight} at (0, ${targetLogicalY})`,
      );
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

  // Load and set Web Search preference
  if (webSearchToggle) {
    try {
      const enabled = await invoke<boolean>("get_enable_web_search");
      webSearchToggle.checked = enabled;
      console.log("Web search preference loaded:", enabled);
    } catch (error) {
      console.error("Failed to load web search preference:", error);
    }

    webSearchToggle.addEventListener("change", async () => {
      try {
        await invoke("set_enable_web_search", { enable: webSearchToggle.checked });
        console.log("Web search preference saved:", webSearchToggle.checked);
        if (settingsStatus) settingsStatus.textContent = "Web search preference saved.";
        // Show checkmark briefly
        if (apiKeyStatusIcon) {
          apiKeyStatusIcon.classList.add("visible");
          setTimeout(() => {
            apiKeyStatusIcon.classList.remove("visible");
            if (settingsStatus && settingsStatus.textContent === "Web search preference saved.") {
              settingsStatus.textContent = "";
            }
          }, 2000);
        } else {
          setTimeout(() => {
            if (settingsStatus && settingsStatus.textContent === "Web search preference saved.") {
              settingsStatus.textContent = "";
            }
          }, 2000);
        }
      } catch (error) {
        console.error("Failed to save web search preference:", error);
        if (settingsStatus) settingsStatus.textContent = "Error saving web search preference.";
      }
    });
  }
}

// --- ADDED: Function to adjust layout based on input area height ---
function updateInputAreaLayout() {
  const inputArea = document.getElementById("input-area");
  const chatHistoryEl = document.getElementById("chat-history");
  const toolButtons = document.getElementById("tool-buttons"); // ADDED
  const containerBottomPadding = 15; // The .container padding-bottom
  const gapBetweenToolButtonsAndInput = 8; // Desired gap

  if (inputArea && chatHistoryEl && toolButtons) {
    // ADDED toolButtons check
    const inputAreaHeight = inputArea.offsetHeight;

    // Position tool-buttons directly above input-area
    toolButtons.style.bottom = `${containerBottomPadding + inputAreaHeight + gapBetweenToolButtonsAndInput}px`;
    const toolButtonsHeight = toolButtons.offsetHeight;

    // Calculate total height occupied by fixed elements at the bottom
    const totalFixedBottomHeight =
      inputAreaHeight + toolButtonsHeight + gapBetweenToolButtonsAndInput;

    const baseChatHistorySpacing = 10; // Base spacing above the topmost fixed element
    chatHistoryEl.style.paddingBottom = `${totalFixedBottomHeight + baseChatHistorySpacing}px`;

    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
  } else if (inputArea && chatHistoryEl) {
    // Fallback if tool-buttons element is not found (e.g. if it were optional)
    const inputAreaHeight = inputArea.offsetHeight;
    const baseSpacing = 15;
    chatHistoryEl.style.paddingBottom = `${inputAreaHeight + baseSpacing}px`;
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
  }
}

// --- Chat Functionality ---
async function addMessageToHistory(
  sender: "You" | "Shard",
  content: string,
  reasoning: string | null = null,
  ocrText: string | null = null,
  imageData?: { base64: string; mime: string } | null,
) {
  console.log(
    `addMessageToHistory called. Sender: ${sender}, Content: ${content}, Reasoning: ${reasoning}, OCR: ${!!ocrText}, Image: ${!!imageData}`,
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
    contentDiv.innerHTML = safeRender(content); // Safe rendering with DOMPurify sanitization
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
    const { accordion, content: sysPromptContent } = createCustomAccordion(
      "Show System Prompt",
      "reasoning",
    );
    accordion.style.marginTop = "10px"; // Add some space above the accordion

    const pre = document.createElement("pre");
    pre.textContent = SYSTEM_INSTRUCTION;
    sysPromptContent.appendChild(pre);

    messageDiv.appendChild(accordion); // Append to the user's message div
  }

  // For user messages: Add Image Accordion if image is present (for Gemini), otherwise OCR Accordion if OCR is present
  if (sender === "You") {
    const selectedModel = modelSelect?.value || ""; // Get current model

    if (imageData) {
      // Image was captured
      if (selectedModel.startsWith("gemini-")) {
        // Gemini model selected
        const imageAccordion = ensureImageAccordion(messageDiv); // New function
        const imageContentDiv = imageAccordion.querySelector(".image-preview-content div");
        if (imageContentDiv) {
          const imgElement = document.createElement("img");
          imgElement.src = `data:${imageData.mime};base64,${imageData.base64}`;
          imgElement.alt = "User uploaded image";
          imgElement.style.maxWidth = "100%"; // Make image responsive within accordion
          imgElement.style.maxHeight = "300px"; // Max height for the preview
          imgElement.style.borderRadius = "4px";
          imgElement.style.objectFit = "contain";
          imageContentDiv.appendChild(imgElement);
        }
      } else {
        // Non-Gemini (OpenRouter) model selected, but image was captured
        if (ocrText) {
          // Show OCR if available
          const ocrAccordion = ensureOcrAccordion(messageDiv);
          const ocrContent = ocrAccordion.querySelector(".ocr-content div");
          if (ocrContent) {
            const pre = document.createElement("pre");
            pre.textContent = ocrText;
            ocrContent.appendChild(pre);
          }
        }
        // No image accordion for OpenRouter models, even if image was captured.
      }
    } else if (ocrText) {
      // No image captured, but OCR text is present
      const ocrAccordion = ensureOcrAccordion(messageDiv);
      const ocrContent = ocrAccordion.querySelector(".ocr-content div");
      if (ocrContent) {
        const pre = document.createElement("pre");
        pre.textContent = ocrText;
        ocrContent.appendChild(pre);
      }
    }
  }

  // Add to chatMessageHistory AFTER it's been decided what to display
  // For streamed messages, this will be updated in STREAM_END
  if (
    sender === "You" ||
    (sender === "Shard" && !chatHistory.querySelector(".message.assistant.thinking"))
  ) {
    const role: "user" | "assistant" = sender === "You" ? "user" : "assistant";
    const historyEntry: ChatMessage = { role, content };
    const selectedModel = modelSelect?.value || ""; // Get current model for history decision

    if (sender === "You") {
      if (imageData && selectedModel.startsWith("gemini-")) {
        // Only add image data to history if a Gemini model is selected
        historyEntry.image_base64_data = imageData.base64;
        historyEntry.image_mime_type = imageData.mime;
        // OCR text is NOT appended to content if an image is present AND Gemini model
      } else if (ocrText) {
        // For OpenRouter models (even if image was captured), or if no image captured,
        // append OCR text to content if available.
        historyEntry.content = content + "\n\nOCR Text: " + ocrText;
      }
    }
    chatMessageHistory.push(historyEntry);
  }

  // Add reasoning accordion if reasoning is present for assistant messages
  if (sender === "Shard" && reasoning) {
    const { accordion, content: reasoningContent } = createCustomAccordion(
      "Show Reasoning",
      "reasoning",
    );
    const pre = document.createElement("pre");
    pre.textContent = reasoning;
    reasoningContent.appendChild(pre);
    messageDiv.appendChild(accordion);
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

// --- Helper to get clean text from contenteditable div ---
function getTextFromContentEditable(div: HTMLDivElement): string {
  let text = "";
  div.childNodes.forEach((node) => {
    if (node.nodeType === Node.TEXT_NODE) {
      text += node.textContent;
    } else if (node.nodeType === Node.ELEMENT_NODE) {
      // Skip our image wrapper, recurse for other elements like <div> or <p> from paste
      if (!(node as HTMLElement).classList?.contains("inline-image-wrapper")) {
        // Basic handling for block elements to add a newline, might need refinement
        if (["DIV", "P", "BR"].includes((node as HTMLElement).tagName)) {
          if (text.length > 0 && !text.endsWith("\\n")) text += "\\n"; // Add newline if not already there
        }
        text += getTextFromContentEditable(node as HTMLDivElement); // Recurse
      }
    }
  });
  return text.trim();
}

// --- Helper functions for clear/undo button state management ---
function updateClearButtonState(isUndoMode: boolean) {
  if (!clearChatButton || !clearIcon || !undoIcon) return;

  console.log(`[UNDO] Updating button state to ${isUndoMode ? "undo" : "clear"} mode`);

  if (isUndoMode) {
    // Switch to undo mode with sequential fade transition
    clearChatButton.title = "Undo Clear";

    // Step 1: Fade out clear icon
    clearIcon.classList.add("fading-out");

    setTimeout(() => {
      // Step 2: Hide clear icon and show undo icon
      clearIcon.classList.add("hidden");
      clearIcon.classList.remove("fading-out");
      undoIcon.classList.remove("hidden");
      undoIcon.classList.add("fading-in");

      setTimeout(() => {
        // Step 3: Fade in undo icon
        undoIcon.classList.remove("fading-in");
      }, 10); // Small delay to ensure DOM update
    }, 100);
  } else {
    // Switch to clear mode with sequential fade transition
    clearChatButton.title = "Clear Chat";

    // Step 1: Fade out undo icon
    undoIcon.classList.add("fading-out");

    setTimeout(() => {
      // Step 2: Hide undo icon and show clear icon
      undoIcon.classList.add("hidden");
      undoIcon.classList.remove("fading-out");
      clearIcon.classList.remove("hidden");
      clearIcon.classList.add("fading-in");

      setTimeout(() => {
        // Step 3: Fade in clear icon
        clearIcon.classList.remove("fading-in");
      }, 10); // Small delay to ensure DOM update
    }, 100);
  }
}

function createChatBackup(): ChatBackup {
  const backup = {
    chatMessageHistory: [...chatMessageHistory], // Deep copy array
    chatHistoryHTML: chatHistory?.innerHTML || "",
    currentOcrText,
    currentImageBase64,
    currentImageMimeType,
    currentTempScreenshotPath,
  };
  console.log("[UNDO] Created backup:", {
    messageCount: backup.chatMessageHistory.length,
    htmlLength: backup.chatHistoryHTML.length,
    hasOcrText: !!backup.currentOcrText,
    hasImage: !!backup.currentImageBase64,
  });
  return backup;
}

function restoreChatFromBackup(backup: ChatBackup) {
  console.log("[UNDO] Restoring backup:", {
    messageCount: backup.chatMessageHistory.length,
    htmlLength: backup.chatHistoryHTML.length,
    hasOcrText: !!backup.currentOcrText,
    hasImage: !!backup.currentImageBase64,
  });
  chatMessageHistory = [...backup.chatMessageHistory]; // Restore array
  if (chatHistory) chatHistory.innerHTML = backup.chatHistoryHTML; // Restore DOM
  currentOcrText = backup.currentOcrText;
  currentImageBase64 = backup.currentImageBase64;
  currentImageMimeType = backup.currentImageMimeType;
  currentTempScreenshotPath = backup.currentTempScreenshotPath;
  console.log("[UNDO] Restored state:", {
    newMessageCount: chatMessageHistory.length,
    htmlRestored: chatHistory?.innerHTML.length || 0,
  });
}

function resetUndoState() {
  console.log("[UNDO] Resetting undo state");
  chatBackup = null;
  canUndo = false;
  updateClearButtonState(false);
}

// --- Combined Clear/Undo Chat Handler ---
async function handleClearOrUndoChat() {
  console.log(
    `[UNDO] handleClearOrUndoChat called - canUndo: ${canUndo}, hasBackup: ${!!chatBackup}`,
  );
  if (canUndo && chatBackup) {
    // Perform undo operation
    console.log("[UNDO] Performing undo operation");
    await undoChatClear();
  } else {
    // Perform clear operation
    console.log("[UNDO] Performing clear operation");
    await clearChatHistory();
  }
}

async function clearChatHistory() {
  // Create backup before clearing
  chatBackup = createChatBackup();

  console.log("Starting fade out for chat clear...");
  chatHistory.classList.add("fade-out");
  chatHistory.classList.remove("fade-in");
  chatHistory.classList.remove("fade-in");

  setTimeout(async () => {
    if (chatHistory) chatHistory.innerHTML = "";
    console.log("Chat history cleared.");
    chatMessageHistory = [];

    // Don't hide the button, just change its state to undo mode
    canUndo = true;
    updateClearButtonState(true);

    if (currentTempScreenshotPath) {
      console.log("Cleanup requested for temp screenshot:", currentTempScreenshotPath);
      try {
        await invoke("cleanup_temp_screenshot", { path: currentTempScreenshotPath });
        console.log("Temp screenshot cleanup successful via clearInlineImageAndData.");
      } catch (err) {
        console.error("Error cleaning up temp screenshot via clearInlineImageAndData:", err);
      }
    }
    // Don't reset state variables here - they're needed for undo
    // Only clear the current variables after backup is created
    currentOcrText = null;
    currentImageBase64 = null;
    currentTempScreenshotPath = null;
    await clearInlineImageAndData(); // Use the new function to clear image if present

    if (statusMessage) statusMessage.textContent = "";

    console.log("Starting fade in after chat clear...");
    chatHistory.classList.remove("fade-out");
    setTimeout(() => {
      chatHistory.classList.remove("fade-in");
      updateInputAreaLayout(); // ADDED: Update layout AFTER fade-in completes
    }, FADE_DURATION_CHATHISTORY);
  }, FADE_DURATION_CHATHISTORY);
}

async function undoChatClear() {
  if (!chatBackup) return;

  console.log("Starting fade out for chat undo...");
  chatHistory.classList.add("fade-out");
  chatHistory.classList.remove("fade-in");

  setTimeout(async () => {
    // Restore from backup
    restoreChatFromBackup(chatBackup!);
    console.log(
      "[UNDO] Chat history restored from backup - messages restored:",
      chatMessageHistory.length,
    );

    // Reset undo state
    resetUndoState();

    if (statusMessage) statusMessage.textContent = "";

    console.log("Starting fade in after chat undo...");
    chatHistory.classList.remove("fade-out");
    setTimeout(() => {
      chatHistory.classList.remove("fade-in");
      updateInputAreaLayout(); // Update layout AFTER fade-in completes
    }, FADE_DURATION_CHATHISTORY);
  }, FADE_DURATION_CHATHISTORY);
}

// --- ADDED: New function to clear inline image and associated data ---
async function clearInlineImageAndData() {
  console.log("clearInlineImageAndData called");
  const imageWrapper = messageInput.querySelector(".inline-image-wrapper");
  if (imageWrapper) {
    imageWrapper.remove();
  }

  if (currentTempScreenshotPath) {
    console.log("Cleanup requested for temp screenshot:", currentTempScreenshotPath);
    try {
      await invoke("cleanup_temp_screenshot", { path: currentTempScreenshotPath });
      console.log("Temp screenshot cleanup successful via clearInlineImageAndData.");
    } catch (err) {
      console.error("Error cleaning up temp screenshot via clearInlineImageAndData:", err);
    }
  }

  currentImageBase64 = null;
  currentImageMimeType = null;
  currentOcrText = null;
  currentTempScreenshotPath = null;
  messageInput.title = ""; // Clear tooltip

  // Ensure the input area is focused after clearing
  messageInput.focus();
  updatePlaceholderState(); // ADDED: Update placeholder
  updateInputAreaLayout(); // Update layout in case height changed
}

// --- ADDED: Function to update placeholder visibility ---
function updatePlaceholderState() {
  if (!messageInput) return;
  const textContent = messageInput.textContent?.trim() || "";
  const hasImage = messageInput.querySelector(".inline-image-wrapper");

  if (textContent === "" && !hasImage) {
    messageInput.classList.add("placeholder-active");
  } else {
    messageInput.classList.remove("placeholder-active");
  }
}

// --- Capture OCR Handler ---
async function handleCaptureOcr() {
  console.log("Capture OCR initiated");
  if (statusMessage) {
    statusMessage.textContent = "Starting screen capture...";
    statusMessage.style.display = "block";
  }
  // Clear previous capture state *before* starting new capture
  await clearInlineImageAndData()

  // Visually indicate loading
  if (ocrIconContainer) ocrIconContainer.style.opacity = "0.5";

  try {
    const result = await core.invoke<CaptureResult>("capture_interactive_and_ocr");
    console.log("Capture Result:", result);

    if (result.image_base64) {
      currentImageBase64 = result.image_base64;
      currentImageMimeType = "image/png"; // Assuming PNG from backend for now, or get from result if available
      currentOcrText = result.ocr_text; // OCR text is still stored
      currentTempScreenshotPath = result.temp_path;

      // Preserve existing text before adding the image
      const existingText = getTextFromContentEditable(messageInput).trim();
      messageInput.innerHTML = ""; // Clear the input field to rebuild

      if (existingText) {
        // Re-add the text. We might need to handle line breaks more explicitly if they were divs/brs
        // For now, let's assume textContent captured the essence.
        // A more robust way might involve iterating childNodes and rebuilding.
        const textNode = document.createTextNode(existingText + " "); // Add a space before the image
        messageInput.appendChild(textNode);
      }

      // Create inline image structure
      const wrapper = document.createElement("span");
      wrapper.className = "inline-image-wrapper";
      wrapper.contentEditable = "false"; // Make the wrapper non-editable

      const img = document.createElement("img");
      img.src = `data:${currentImageMimeType};base64,${currentImageBase64}`;
      img.alt = "Captured image";

      // ADDED: Update layout *after* the image has loaded and has dimensions
      img.onload = () => {
        updateInputAreaLayout();
      };
      // Add data attributes if needed for other purposes, e.g., temp path
      if (currentTempScreenshotPath) {
        img.setAttribute("data-temp-path", currentTempScreenshotPath);
      }

      const deleteBtn = document.createElement("button");
      deleteBtn.className = "delete-inline-image-btn";
      deleteBtn.innerHTML = "&times;"; // '×' character
      deleteBtn.title = "Remove image";
      deleteBtn.addEventListener("click", async (e) => {
        e.stopPropagation(); // Prevent triggering other input events
        await clearInlineImageAndData();
      });

      wrapper.appendChild(img);
      wrapper.appendChild(deleteBtn);
      messageInput.appendChild(wrapper);

      // Add a non-breaking space after the image to allow typing after it
      // Or set focus appropriately. For now, let's just append.
      // The user can click after the image to type.
      // Or, better, programmatically set the selection:
      const range = document.createRange();
      const sel = window.getSelection();
      range.setStartAfter(wrapper);
      range.collapse(true);
      sel?.removeAllRanges();
      sel?.addRange(range);

      updatePlaceholderState(); // ADDED: Update placeholder after adding image

      if (currentOcrText) {
        messageInput.title = currentOcrText; // Set tooltip
      }
    } else if (result.ocr_text) {
      // No image, but OCR text is present. Append it or replace if input is empty.
      const existingText = getTextFromContentEditable(messageInput).trim();
      if (existingText) {
        messageInput.textContent = existingText + "\n" + result.ocr_text; // Append with a newline
      } else {
        messageInput.textContent = result.ocr_text; // Replace if empty
      }
      currentOcrText = result.ocr_text;
      messageInput.title = currentOcrText;
      // Place cursor at the end of the text
      const range = document.createRange();
      const sel = window.getSelection();
      if (messageInput.lastChild) {
        range.setStartAfter(messageInput.lastChild);
      } else {
        range.selectNodeContents(messageInput);
        range.collapse(false);
      }
      sel?.removeAllRanges();
      sel?.addRange(range);

      updatePlaceholderState(); // ADDED: Update placeholder after adding OCR text
    }

    if (statusMessage) {
      if (currentImageBase64 || currentOcrText) {
        statusMessage.textContent = currentImageBase64 ? "Image captured." : "OCR text captured.";
        setTimeout(() => {
          if (
            statusMessage &&
            (statusMessage.textContent === "Image captured." ||
              statusMessage.textContent === "OCR text captured.")
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
    await clearInlineImageAndData(); // Clear any partial state on error

    if (statusMessage) {
      statusMessage.textContent = `Error: ${errorMessage}`;
      setTimeout(() => {
        if (statusMessage && statusMessage.textContent === `Error: ${errorMessage}`) {
          statusMessage.style.display = "none";
          statusMessage.textContent = "";
        }
      }, 5000);
    }
  } finally {
    if (ocrIconContainer) ocrIconContainer.style.opacity = "1.0";
    messageInput.focus();
    updateInputAreaLayout();
  }
}

// --- Send Message Handler ---
async function handleSendMessage() {
  if (isAIResponding) {
    console.log(
      "handleSendMessage: AI is responding. Finalizing current response before interjection.",
    );
    finalizeCurrentResponse();
  }

  const userTypedText = getTextFromContentEditable(messageInput);
  let textToDisplay = userTypedText;
  let imagePayloadForMessage: { base64: string; mime: string } | null = null;
  let tempOcrTextForAccordion: string | null = currentOcrText;

  if (currentImageBase64 && currentImageMimeType) {
    imagePayloadForMessage = { base64: currentImageBase64, mime: currentImageMimeType };
    if (!userTypedText) {
      textToDisplay = "[Image sent]";
    }
  } else if (!userTypedText) {
    console.log("handleSendMessage: No text typed and no image.");
    return;
  }

  addMessageToHistory("You", textToDisplay, null, tempOcrTextForAccordion, imagePayloadForMessage);

  const messagesToSendToBackend = [...chatMessageHistory];

  await clearInlineImageAndData();
  messageInput.innerHTML = "";
  messageInput.title = "";
  messageInput.contentEditable = "false"; // ADDED: Disable input during AI response
  updatePlaceholderState();

  // Reset undo state when sending new message
  resetUndoState();

  // Show clear button if it was hidden
  if (clearChatButton?.classList.contains("hidden")) {
    clearChatButton.classList.remove("hidden");
  }

  // Clean up any leftover streaming dots from all responses before starting new one
  console.log(`[DEBUG] Cleaning up leftover streaming dots before new response`);
  responseDivMap.forEach((divs, counter) => {
    const leftoverDots = divs.contentDiv.querySelectorAll(".streaming-dots");
    if (leftoverDots.length > 0) {
      console.log(
        `[DEBUG] Found ${leftoverDots.length} leftover streaming dots in response ${counter}`,
      );
      leftoverDots.forEach((dots, index) => {
        dots.remove();
        console.log(
          `[DEBUG] Removed leftover streaming dots ${index + 1} from response ${counter}`,
        );
      });
    }
  });

  // Show thinking indicator / initial placeholder for Shard's response
  console.log(`[DEBUG] Creating new assistant message div for response ${responseCounter}`);
  const assistantMessagePlaceholder = document.createElement("div");
  assistantMessagePlaceholder.classList.add("message", "assistant", "streaming"); // New class for styling streamed message
  assistantMessagePlaceholder.setAttribute("data-response-counter", responseCounter.toString());
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
  console.log(
    `[DEBUG] Set currentAssistantMessageDiv for response ${responseCounter}, element:`,
    currentAssistantMessageDiv,
  );
  console.log(
    `[DEBUG] Set currentAssistantContentDiv for response ${responseCounter}, element:`,
    currentAssistantContentDiv,
  );

  // Store in map for this specific response - increment counter first
  responseCounter++; // Increment counter for new response
  console.log(`[DEBUG] Starting new response with counter: ${responseCounter}`);
  responseDivMap.set(responseCounter, {
    messageDiv: assistantMessagePlaceholder,
    contentDiv: currentAssistantContentDiv,
  });

  updateInputAreaLayout();

  if (chatHistory) {
    chatHistory.appendChild(assistantMessagePlaceholder);
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  isAIResponding = true; // Set flag as AI is about to respond
  streamFinalized = false; // Reset stream finalized flag for new response
  await setupStreamListeners(); // Re-setup listeners for new response counter
  try {
    // Invoke send_text_to_model. It no longer directly returns the message content.
    await core.invoke("send_text_to_model", {
      messages: messagesToSendToBackend,
      window: Window.getCurrent(), // Pass the current window
      // ADDED: Pass the current image data if available, even if also in messages.
      // The backend can decide how to use it (e.g., for Gemini File API vs inline base64).
      // This simplifies frontend logic slightly. The backend's ChatMessage struct
      // already supports image_base64_data and image_mime_type.
      // The send_text_to_model will primarily look at the messages array.
      // The crucial part is that chatMessageHistory entries for "user" type
      // (created by addMessageToHistory) should contain the image data
      // if an image was present for that user message.
    });
    console.log("send_text_to_model invoked. Waiting for stream events.");

    // The rest of the logic (removeThinkingIndicator, addMessageToHistory for Shard)
    // will now be handled by the STREAM_CHUNK, STREAM_END, and STREAM_ERROR event listeners.
  } catch (error) {
    console.error("Failed to invoke send_text_to_model:", error);
    if (currentAssistantMessageDiv) {
      currentAssistantMessageDiv.classList.remove("streaming");
      currentAssistantMessageDiv.classList.add("error");
    }
    isAIResponding = false;
    await clearInlineImageAndData(); // Clear image data on error too
  } finally {
    updateInputAreaLayout();
  }
}

// --- Event Listeners ---

// Function to ensure reasoning accordion exists
function ensureReasoningAccordion(messageDiv: HTMLElement) {
  let reasoningAccordion = messageDiv.querySelector(".reasoning-accordion") as HTMLElement;

  if (!reasoningAccordion) {
    const { accordion, toggle, content } = createCustomAccordion("Reasoning", "reasoning");
    toggle.setAttribute("aria-expanded", "true"); // Start open
    content.classList.add("open"); // Start open

    const reasoningDiv = document.createElement("div");
    content.appendChild(reasoningDiv);

    reasoningAccordion = accordion;

    // Insert after message content
    const messageContent = messageDiv.querySelector(".message-content");
    if (messageContent && messageContent.nextSibling) {
      messageDiv.insertBefore(reasoningAccordion, messageContent.nextSibling);
    } else {
      messageDiv.appendChild(reasoningAccordion);
    }
  }

  return reasoningAccordion;
}

function ensureOcrAccordion(messageDiv: HTMLElement) {
  let ocrAccordion = messageDiv.querySelector(".ocr-accordion") as HTMLElement;

  if (!ocrAccordion) {
    const { accordion, toggle, content } = createCustomAccordion("OCR Text", "ocr");
    toggle.setAttribute("aria-expanded", "false"); // Start closed
    // content starts closed by default

    const ocrDiv = document.createElement("div");
    content.appendChild(ocrDiv);

    ocrAccordion = accordion;

    // Insert after message content but before reasoning accordion if it exists
    const messageContent = messageDiv.querySelector(".message-content");
    const reasoningAccordion = messageDiv.querySelector(".reasoning-accordion");

    if (reasoningAccordion) {
      messageDiv.insertBefore(ocrAccordion, reasoningAccordion);
    } else if (messageContent && messageContent.nextSibling) {
      messageDiv.insertBefore(ocrAccordion, messageContent.nextSibling);
    } else {
      messageDiv.appendChild(ocrAccordion);
    }
  }

  return ocrAccordion;
}

// NEW function to ensure Image Preview accordion exists
function ensureImageAccordion(messageDiv: HTMLElement): HTMLElement {
  let imageAccordion = messageDiv.querySelector(".image-preview-accordion") as HTMLElement;

  if (!imageAccordion) {
    const { accordion, toggle, content } = createCustomAccordion("Image Preview", "image-preview"); // New type
    toggle.setAttribute("aria-expanded", "true"); // Start open by default for images
    content.classList.add("open");

    const imagePreviewDiv = document.createElement("div"); // This div will hold the image
    content.appendChild(imagePreviewDiv);

    imageAccordion = accordion;

    // Insert after message content, but before OCR or Reasoning if they exist
    const messageContent = messageDiv.querySelector(".message-content");
    const ocrAccordionElement = messageDiv.querySelector(".ocr-accordion");
    const reasoningAccordionElement = messageDiv.querySelector(".reasoning-accordion");

    if (ocrAccordionElement) {
      messageDiv.insertBefore(imageAccordion, ocrAccordionElement);
    } else if (reasoningAccordionElement) {
      messageDiv.insertBefore(imageAccordion, reasoningAccordionElement);
    } else if (messageContent && messageContent.nextSibling) {
      messageDiv.insertBefore(imageAccordion, messageContent.nextSibling);
    } else {
      messageDiv.appendChild(imageAccordion);
    }
  }
  return imageAccordion;
}

// Function to finalize current AI response when interjected
function finalizeCurrentResponse() {
  if (currentAssistantMessageDiv && currentAssistantContentDiv && isAIResponding) {
    console.log("Finalizing current AI response due to interjection");
    console.log(
      `[DEBUG] Finalizing response ${responseCounter}, currentAssistantContentDiv content: "${currentAssistantContentDiv.textContent}"`,
    );

    // Signal backend to cancel current stream
    core.invoke("cancel_current_stream").catch((error) => {
      console.warn("Failed to cancel backend stream:", error);
    });

    // Get the current response's content div
    const currentResponseContentDiv = responseDivMap.get(responseCounter)?.contentDiv;

    // Get the current content that was streamed so far
    const currentContent = currentResponseContentDiv?.textContent || "";

    // ALWAYS add an assistant message to chat history to maintain conversation flow
    // This prevents consecutive user messages which confuse the model
    const interruptedContent = currentContent.trim()
      ? currentContent + " [interrupted]"
      : "[interrupted]";

    chatMessageHistory.push({
      role: "assistant",
      content: interruptedContent,
    });

    // Remove thinking indicator if present
    if (currentAssistantMessageDiv.classList.contains("thinking")) {
      currentAssistantMessageDiv.classList.remove("thinking");
      const dotsContainer = currentAssistantMessageDiv.querySelector(".dots-container");
      if (dotsContainer) {
        dotsContainer.remove();
      }
    }

    // Remove ALL streaming dots from the current response's content div
    if (currentResponseContentDiv) {
      const allStreamingDots = currentResponseContentDiv.querySelectorAll(".streaming-dots");
      console.log(`[DEBUG] Finalizing: Found ${allStreamingDots.length} streaming dots to remove`);
      allStreamingDots.forEach((dots, index) => {
        dots.remove();
        console.log(`[DEBUG] Finalizing: Removed streaming dots ${index + 1}`);
      });

      // Add "[interrupted]" indicator to show the response was cut off
      const interruptedIndicator = document.createElement("span");
      interruptedIndicator.textContent = " [interrupted]";
      interruptedIndicator.style.color = "rgba(255, 255, 255, 0.5)";
      interruptedIndicator.style.fontSize = "0.85em";
      currentResponseContentDiv.appendChild(interruptedIndicator);
    }

    // Reset streaming state but keep the response div map intact
    console.log(
      `[DEBUG] Resetting streaming state - setting currentAssistantMessageDiv and currentAssistantContentDiv to null`,
    );
    currentAssistantMessageDiv = null;
    currentAssistantContentDiv = null;
    isAIResponding = false;

    // Clear any pending stream buffers and animation state
    streamDeltaBuffer = "";
    if (streamAnimationFrameRequested) {
      streamAnimationFrameRequested = false;
    }
    streamFinalized = false; // Reset for next response
  }
}

// Define interfaces for stream payloads
interface StreamChunkPayload {
  content?: string | null;
  role?: string | null;
  reasoning?: string | null;
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
let unlistenArticleLookupStarted: (() => void) | null = null;
let unlistenArticleLookupCompleted: (() => void) | null = null;
let unlistenWeatherLookupStarted: (() => void) | null = null;
let unlistenWeatherLookupCompleted: (() => void) | null = null;
let unlistenFinancialDataStarted: (() => void) | null = null;
let unlistenFinancialDataCompleted: (() => void) | null = null;
let unlistenArxivLookupStarted: (() => void) | null = null; // ADDED
let unlistenArxivLookupCompleted: (() => void) | null = null; // ADDED

// Buffer and flag for batched animation of stream chunks
let streamDeltaBuffer = "";
let streamAnimationFrameRequested = false;
let streamFinalized = false; // Flag to prevent further modifications after STREAM_END

const MAX_SUB_CHUNK_LENGTH = 70;
const SUB_CHUNK_ANIMATION_DELAY = 50;

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

// --- Functions to create icons ---
function createGlobeIcon(): SVGSVGElement {
  const svgNS = "http://www.w3.org/2000/svg";
  const icon = document.createElementNS(svgNS, "svg");
  icon.setAttribute("width", "18");
  icon.setAttribute("height", "18");
  icon.setAttribute("viewBox", "0 0 24 24");
  icon.setAttribute("fill", "none");
  icon.setAttribute("stroke", "currentColor");
  icon.setAttribute("stroke-width", "2");
  icon.setAttribute("stroke-linecap", "round");
  icon.setAttribute("stroke-linejoin", "round");
  // Apply base classes for styling and Lucide identity. Spinning is now opt-in.
  icon.classList.add("lucide", "lucide-globe", "web-search-globe-icon");

  const circle1 = document.createElementNS(svgNS, "circle");
  circle1.setAttribute("cx", "12");
  circle1.setAttribute("cy", "12");
  circle1.setAttribute("r", "10");
  icon.appendChild(circle1);

  const path1 = document.createElementNS(svgNS, "path");
  path1.setAttribute("d", "M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20");
  icon.appendChild(path1);

  const path2 = document.createElementNS(svgNS, "path");
  path2.setAttribute("d", "M2 12h20");
  icon.appendChild(path2);

  return icon;
}

function createThermometerIcon(): SVGSVGElement {
  const svgNS = "http://www.w3.org/2000/svg";
  const icon = document.createElementNS(svgNS, "svg");
  icon.setAttribute("width", "18");
  icon.setAttribute("height", "18");
  icon.setAttribute("viewBox", "0 0 24 24");
  icon.setAttribute("fill", "none");
  icon.setAttribute("stroke", "currentColor");
  icon.setAttribute("stroke-width", "2");
  icon.setAttribute("stroke-linecap", "round");
  icon.setAttribute("stroke-linejoin", "round");
  icon.classList.add("lucide", "lucide-thermometer-icon", "lucide-thermometer");
  const path = document.createElementNS(svgNS, "path");
  path.setAttribute("d", "M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z");
  icon.appendChild(path);
  return icon;
}

function createFinancialIcon(): SVGSVGElement {
  const svgNS = "http://www.w3.org/2000/svg";
  const icon = document.createElementNS(svgNS, "svg");
  icon.setAttribute("width", "18");
  icon.setAttribute("height", "18");
  icon.setAttribute("viewBox", "0 0 24 24");
  icon.setAttribute("fill", "none");
  icon.setAttribute("stroke", "currentColor");
  icon.setAttribute("stroke-width", "2");
  icon.setAttribute("stroke-linecap", "round");
  icon.setAttribute("stroke-linejoin", "round");
  icon.classList.add("lucide", "lucide-dollar-sign"); // Or appropriate Lucide class
  const path1 = document.createElementNS(svgNS, "path");
  path1.setAttribute("d", "M12 1v22M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"); // Example path
  icon.appendChild(path1);
  return icon;
}

function createArxivIcon(): SVGSVGElement {
  const svgNS = "http://www.w3.org/2000/svg";
  const icon = document.createElementNS(svgNS, "svg");
  icon.setAttribute("width", "24");
  icon.setAttribute("height", "24");
  icon.setAttribute("viewBox", "0 0 24 24");
  icon.setAttribute("fill", "none");
  icon.setAttribute("stroke", "currentColor");
  icon.setAttribute("stroke-width", "2");
  icon.setAttribute("stroke-linecap", "round");
  icon.setAttribute("stroke-linejoin", "round");
  icon.classList.add("lucide", "lucide-book-icon", "lucide-book"); // Using book icon

  const path1 = document.createElementNS(svgNS, "path");
  path1.setAttribute("d", "M4 19.5A2.5 2.5 0 0 1 6.5 17H20");
  icon.appendChild(path1);
  const path2 = document.createElementNS(svgNS, "path");
  path2.setAttribute("d", "M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z");
  icon.appendChild(path2);

  return icon;
}

// --- Define interfaces for Article Lookup event payloads --- (Matches backend)
interface ArticleLookupStartedPayload {
  query: string;
}
interface ArticleLookupCompletedPayload {
  query: string;
  success: boolean;
  summary?: string | null;
  source_name?: string[] | null;
  source_url?: string[] | null;
  error?: string | null;
}

// --- ADDED: Define interfaces for Weather Lookup event payloads --- (Matches backend)
interface WeatherLookupStartedPayload {
  location: string;
}
interface WeatherLookupCompletedPayload {
  location: string;
  success: boolean;
  temperature?: number | null;
  unit?: string | null;
  description?: string | null;
  error?: string | null;
}

// Ensure these are present:
interface FinancialDataStartedPayload {
  query: string;
  symbol: string;
}
interface FinancialDataCompletedPayload {
  query: string;
  symbol: string;
  success: boolean;
  data?: string | null;
  error?: string | null;
}

// --- ADDED: Define interfaces for ArXiv Lookup event payloads --- (Matches backend)
interface ArxivPaperSummary {
  title: string;
  summary: string;
  authors: string[];
  id: string;
  published_date?: string | null;
  pdf_url: string;
}

interface ArxivLookupStartedPayload {
  query: string;
}

interface ArxivLookupCompletedPayload {
  query: string;
  success: boolean;
  results?: ArxivPaperSummary[] | null;
  error?: string | null;
}

async function setupStreamListeners() {
  if (unlistenStreamChunk) unlistenStreamChunk();
  const listenerResponseCounter = responseCounter; // Capture current response counter
  console.log(
    `[DEBUG] Setting up stream listeners for response counter: ${listenerResponseCounter}`,
  );
  unlistenStreamChunk = await listen<StreamChunkPayload>("STREAM_CHUNK", (event) => {
    console.log(
      `[DEBUG] STREAM_CHUNK received - listener counter: ${listenerResponseCounter}, current counter: ${responseCounter}, content: "${event.payload.content}"`,
    );

    // Get the content div for this specific response
    const responseDivs = responseDivMap.get(listenerResponseCounter);
    const responseContentDiv = responseDivs?.contentDiv;
    console.log(
      `[DEBUG] responseContentDiv exists for counter ${listenerResponseCounter}:`,
      !!responseContentDiv,
    );

    // Always process events for their own response, don't ignore based on counter mismatch
    // This allows cancelled streams to still update their own message divs

    // Handle content from the new payload structure
    if (event.payload.content && responseContentDiv) {
      streamDeltaBuffer += event.payload.content;
      console.log(`[DEBUG] Added to streamDeltaBuffer, total length: ${streamDeltaBuffer.length}`);
    }

    // Handle reasoning data
    if (event.payload.reasoning && responseDivs?.messageDiv) {
      ensureReasoningAccordion(responseDivs.messageDiv);
      const reasoningAccordion = responseDivs.messageDiv.querySelector(".reasoning-accordion");
      const reasoningContent = reasoningAccordion?.querySelector(".reasoning-content div");
      if (reasoningContent) {
        const currentContent = reasoningContent.getAttribute("data-raw-content") || "";
        const updatedContent = currentContent + event.payload.reasoning;
        reasoningContent.setAttribute("data-raw-content", updatedContent);
        reasoningContent.innerHTML = safeRender(updatedContent);
      }
    }

    if (!streamAnimationFrameRequested && responseContentDiv && !streamFinalized) {
      streamAnimationFrameRequested = true;
      requestAnimationFrame(() => {
        if (!responseContentDiv || streamFinalized) {
          // Double check in case it became null or stream was finalized
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
          if (responseContentDiv.innerHTML.includes("dots-container")) {
            // This will likely be false now
            responseContentDiv.innerHTML = ""; // Clear initial thinking dots (if they were the old style)
          }

          // Remove any existing streaming dots before adding new text
          const existingDots = responseContentDiv.querySelector(".streaming-dots");
          if (existingDots) {
            existingDots.remove();
          }

          // Function to animate text piece by piece
          function animateTextSequentially(textToProcess: string) {
            if (!textToProcess || !responseContentDiv || streamFinalized) return;

            const subChunk = textToProcess.substring(0, MAX_SUB_CHUNK_LENGTH);
            const remainingText = textToProcess.substring(MAX_SUB_CHUNK_LENGTH);

            const newSpan = document.createElement("span");
            newSpan.innerHTML = safeRenderInline(subChunk); // Safe rendering with DOMPurify
            newSpan.style.opacity = "0";
            newSpan.style.transition = "opacity 0.3s ease-out";
            responseContentDiv.appendChild(newSpan);

            requestAnimationFrame(() => {
              // Fade in this piece
              newSpan.style.opacity = "1";
            });

            if (chatHistory) {
              chatHistory.scrollTop = chatHistory.scrollHeight;
            }

            if (remainingText && !streamFinalized) {
              setTimeout(() => {
                animateTextSequentially(remainingText);
              }, SUB_CHUNK_ANIMATION_DELAY);
            } else if (!streamFinalized) {
              // Append streaming dots after the last sub-chunk is animated
              if (responseContentDiv) {
                responseContentDiv.appendChild(getStreamingDots());
                if (chatHistory) chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll again after adding dots
              }
            }
          }
          animateTextSequentially(currentBatchText); // Start processing the batch
        } else if (
          responseContentDiv.innerHTML !== "" &&
          !responseContentDiv.querySelector(".streaming-dots")
        ) {
          // If buffer was empty but there's content and no dots, add dots (e.g. after clearing initial dots)
          responseContentDiv.appendChild(getStreamingDots());
          if (chatHistory) chatHistory.scrollTop = chatHistory.scrollHeight;
        }
      });
    } else if (!responseContentDiv && streamDeltaBuffer) {
      console.warn(
        "STREAM_CHUNK: responseContentDiv is null, but deltaBuffer has content:",
        streamDeltaBuffer,
      );
      streamDeltaBuffer = "";
      streamAnimationFrameRequested = false;
    }
  });

  if (unlistenArticleLookupStarted) unlistenArticleLookupStarted();
  unlistenArticleLookupStarted = await listen<ArticleLookupStartedPayload>(
    "ARTICLE_LOOKUP_STARTED",
    (event) => {
      console.log("ARTICLE_LOOKUP_STARTED received:", event.payload);
      const currentResponseContentDiv = responseDivMap.get(responseCounter)?.contentDiv;
      if (currentResponseContentDiv) {
        // Remove any previous article lookup status ONLY
        const existingStatus = currentResponseContentDiv.querySelector(
          ".article-lookup-status-container",
        );
        if (existingStatus) {
          existingStatus.remove();
        }
        // Also remove general streaming dots if they are the only thing
        const existingDots = currentResponseContentDiv.querySelector(".streaming-dots");
        if (
          existingDots &&
          currentResponseContentDiv.children.length === 1 &&
          currentResponseContentDiv.firstChild === existingDots
        ) {
          existingDots.remove();
        }

        const lookupStatusDiv = document.createElement("div");
        lookupStatusDiv.classList.add("article-lookup-status-container");

        const globeIcon = createGlobeIcon();
        globeIcon.classList.add("spinning-globe");
        lookupStatusDiv.appendChild(globeIcon);

        const statusText = document.createElement("span");
        statusText.textContent = `Looking up article for: "${event.payload.query}"...`;
        statusText.classList.add("article-lookup-status-text");
        lookupStatusDiv.appendChild(statusText);

        currentResponseContentDiv.insertBefore(
          lookupStatusDiv,
          currentResponseContentDiv.firstChild,
        );
        if (chatHistory) chatHistory.scrollTop = chatHistory.scrollHeight;
      }
    },
  );

  if (unlistenArticleLookupCompleted) unlistenArticleLookupCompleted();
  unlistenArticleLookupCompleted = await listen<ArticleLookupCompletedPayload>(
    "ARTICLE_LOOKUP_COMPLETED",
    (event) => {
      console.log("ARTICLE_LOOKUP_COMPLETED received:", event.payload);
      const currentResponseContentDiv = responseDivMap.get(responseCounter)?.contentDiv;
      if (currentResponseContentDiv) {
        const searchingStatusContainer = currentResponseContentDiv.querySelector(
          ".article-lookup-status-container",
        );
        if (searchingStatusContainer) {
          searchingStatusContainer.remove();
        }

        if (event.payload.success && event.payload.summary) {
          const { accordion, toggle, content } = createCustomAccordion(
            `Wikipedia Results: "${event.payload.query}"`,
            "web-search",
          );

          const globeIcon = createGlobeIcon();
          addIconToToggle(toggle, globeIcon);

          const searchContentDiv = content;

          if (event.payload.source_name || event.payload.source_url) {
            const sourceInfo = document.createElement("p");
            sourceInfo.classList.add("web-search-source-info");
            sourceInfo.appendChild(document.createTextNode("Sources: "));

            const payload_source_names = event.payload.source_name; // string[] | undefined
            const payload_source_urls = event.payload.source_url; // string[] | undefined

            console.log("Lengths:", payload_source_names?.length, payload_source_urls?.length);

            if (
              payload_source_names &&
              payload_source_urls &&
              payload_source_names.length > 0 &&
              payload_source_names.length === payload_source_urls.length
            ) {
              console.log(
                "Source names and URLs available:",
                payload_source_names,
                payload_source_urls,
              );
              payload_source_names.forEach((name, index) => {
                const url = payload_source_urls[index];
                const sourceLink = document.createElement("a");
                sourceLink.href = url;
                sourceLink.textContent = name;
                sourceLink.target = "_blank";
                sourceLink.rel = "noreferrer";
                sourceInfo.appendChild(sourceLink);
                if (index < payload_source_names.length - 1) {
                  sourceInfo.appendChild(document.createTextNode(", "));
                }
              });
            } else if (payload_source_names && payload_source_names.length > 0) {
              // Only names available
              sourceInfo.appendChild(document.createTextNode(payload_source_names.join(", ")));
            } else if (payload_source_urls && payload_source_urls.length > 0) {
              // Only URLs available
              payload_source_urls.forEach((url, index) => {
                const sourceLink = document.createElement("a");
                sourceLink.href = url;
                sourceLink.textContent = url; // Use URL as text if no name
                sourceLink.target = "_blank";
                sourceLink.rel = "noopener noreferrer";
                sourceInfo.appendChild(sourceLink);
                if (index < payload_source_urls.length - 1) {
                  sourceInfo.appendChild(document.createTextNode(", "));
                }
              });
            } else {
              // Fallback if no source names or URLs are provided in the payload
              // If you prefer "N/A" when payloaCompare France and Germanyd sources are empty:
              sourceInfo.appendChild(document.createTextNode("Sources: N/A"));
            }
            searchContentDiv.appendChild(sourceInfo);
          }

          if (event.payload.summary) {
            const summaryDiv = document.createElement("div");
            summaryDiv.classList.add("web-search-answer");
            summaryDiv.textContent = cleanLatexMarkup(event.payload.summary);
            searchContentDiv.appendChild(summaryDiv);
          }

          currentResponseContentDiv.insertBefore(accordion, currentResponseContentDiv.firstChild);
        } else if (event.payload.error) {
          console.error("Article lookup failed:", event.payload.error);
        }

        if (
          currentResponseContentDiv.innerHTML === "" ||
          (event.payload.success &&
            currentResponseContentDiv.children.length === 1 &&
            currentResponseContentDiv.querySelector(".web-search-accordion"))
        ) {
          currentResponseContentDiv.appendChild(getStreamingDots());
        }
        if (chatHistory) chatHistory.scrollTop = chatHistory.scrollHeight;
      }
    },
  );

  if (unlistenFinancialDataStarted) unlistenFinancialDataStarted();
  unlistenFinancialDataStarted = await listen<FinancialDataStartedPayload>(
    "FINANCIAL_DATA_STARTED",
    (event) => {
      console.log("FINANCIAL_DATA_STARTED received:", event.payload);
      const currentResponseContentDiv = responseDivMap.get(responseCounter)?.contentDiv;
      if (currentResponseContentDiv) {
        // Remove any previous financial data status ONLY
        const existingStatus = currentResponseContentDiv.querySelector(
          ".financial-data-status-container",
        );
        if (existingStatus) {
          existingStatus.remove();
        }
        // Also remove general streaming dots if they are the only thing
        const existingDots = currentResponseContentDiv.querySelector(".streaming-dots");
        if (
          existingDots &&
          currentResponseContentDiv.children.length === 1 &&
          currentResponseContentDiv.firstChild === existingDots
        ) {
          existingDots.remove();
        }

        const statusDiv = document.createElement("div");
        statusDiv.classList.add("financial-data-status-container");

        const globeIcon = createFinancialIcon(); // Using globe, update if you have a specific financial icon
        globeIcon.classList.add("spinning-globe");
        statusDiv.appendChild(globeIcon);

        const statusText = document.createElement("span");
        statusText.textContent = `Fetching financial data for: "${event.payload.symbol}"...`;
        statusText.classList.add("financial-data-status-text");
        statusDiv.appendChild(statusText);

        // Prepend the new status
        currentResponseContentDiv.insertBefore(statusDiv, currentResponseContentDiv.firstChild);
        if (chatHistory) chatHistory.scrollTop = chatHistory.scrollHeight;
      }
    },
  );

  if (unlistenFinancialDataCompleted) unlistenFinancialDataCompleted();
  unlistenFinancialDataCompleted = await listen<FinancialDataCompletedPayload>(
    "FINANCIAL_DATA_COMPLETED",
    (event) => {
      console.log("FINANCIAL_DATA_COMPLETED received:", event.payload);
      const currentResponseContentDiv = responseDivMap.get(responseCounter)?.contentDiv;
      if (currentResponseContentDiv) {
        const fetchingStatusContainer = currentResponseContentDiv.querySelector(
          ".financial-data-status-container",
        );
        if (fetchingStatusContainer) {
          fetchingStatusContainer.remove();
        }

        const { accordion, toggle, content } = createCustomAccordion(
          `Financial Data for: "${event.payload.symbol}"`,
          "web-search",
        );

        const financialIcon = createFinancialIcon();
        addIconToToggle(toggle, financialIcon);

        const financialContentDiv = content;
        financialContentDiv.classList.add("financial-content");

        if (event.payload.success && event.payload.symbol && event.payload.data) {
          // Success with data
          const dataDiv = document.createElement("div");
          dataDiv.classList.add("financial-data-text");
          dataDiv.textContent = event.payload.data;
          financialContentDiv.appendChild(dataDiv);
        } else {
          // Error or No Data case
          const errorParagraph = document.createElement("p");
          errorParagraph.classList.add("financial-lookup-error-text"); // For styling if needed

          if (event.payload.error) {
            console.error("Financial data lookup error:", event.payload.error);
            errorParagraph.textContent = `Financial data lookup for "${event.payload.symbol}" failed: ${event.payload.error}.`;
          } else if (event.payload.success && !event.payload.data && event.payload.symbol) {
            // Success, but no specific financial data
            errorParagraph.textContent = `No specific financial data found for "${event.payload.symbol}".`;
          } else {
            // General fallback
            errorParagraph.textContent = `An unexpected issue occurred while fetching financial data for "${event.payload.symbol}".`;
          }
          financialContentDiv.appendChild(errorParagraph);

          const tipParagraph = document.createElement("p");
          tipParagraph.classList.add("financial-lookup-tip"); // For styling if needed
          tipParagraph.textContent =
            "If the stock isn't found, try the name capitalized or the symbol in all caps!";
          financialContentDiv.appendChild(tipParagraph);

          // Optionally open the accordion if there's an error/tip
          toggle.setAttribute("aria-expanded", "true");
          content.classList.add("open");
        }

        currentResponseContentDiv.insertBefore(accordion, currentResponseContentDiv.firstChild);

        // Ensure streaming dots are present if no other content is being streamed by the LLM yet
        if (
          !currentResponseContentDiv.querySelector(".streaming-dots") &&
          (currentResponseContentDiv.innerHTML === "" || // Empty
            (currentResponseContentDiv.children.length > 0 && // Only has accordions/tool messages
              currentResponseContentDiv.querySelectorAll(
                ":not(.streaming-dots):not(.web-search-accordion):not(.tool-error-message):not(.tool-info-message)",
              ).length === 0))
        ) {
          currentResponseContentDiv.appendChild(getStreamingDots());
        }
        if (chatHistory) chatHistory.scrollTop = chatHistory.scrollHeight;
      }
    },
  );

  if (unlistenStreamEnd) unlistenStreamEnd();
  unlistenStreamEnd = await listen<StreamEndPayload>("STREAM_END", async (event) => {
    console.log(
      `[DEBUG] STREAM_END received - listener counter: ${listenerResponseCounter}, current counter: ${responseCounter}`,
    );
    console.log(`[DEBUG] STREAM_END content: "${event.payload.full_content}"`);

    // Get the divs for this specific response
    const responseDivs = responseDivMap.get(listenerResponseCounter);
    const responseMessageDiv = responseDivs?.messageDiv;
    const responseContentDiv = responseDivs?.contentDiv;

    console.log(
      `[DEBUG] responseMessageDiv exists for counter ${listenerResponseCounter}:`,
      !!responseMessageDiv,
    );
    console.log(
      `[DEBUG] responseContentDiv exists for counter ${listenerResponseCounter}:`,
      !!responseContentDiv,
    );

    console.log("STREAM_END received:", event.payload);
    console.log(`[DEBUG] responseDivs from map:`, responseDivs);
    console.log(`[DEBUG] Available response counters in map:`, Array.from(responseDivMap.keys()));

    if (responseMessageDiv && responseContentDiv) {
      // Remove ALL streaming dots from this response's content div
      const allStreamingDots = responseContentDiv.querySelectorAll(".streaming-dots");
      console.log(`[DEBUG] Found ${allStreamingDots.length} streaming dots to remove`);
      allStreamingDots.forEach((dots, index) => {
        dots.remove();
        console.log(
          `[DEBUG] Removed streaming dots ${index + 1} from response ${listenerResponseCounter}`,
        );
      });

      // Guard the DOM operations specifically
      if (responseContentDiv) {
        // First, save any accordions that might be present
        const accordions: { element: HTMLElement; type: string }[] = [];
        responseContentDiv.querySelectorAll(".web-search-accordion").forEach((accordionNode) => {
          const accordionElement = accordionNode as HTMLElement;
          let type = "article";
          if (accordionElement.querySelector(".weather-info-text")) type = "weather";
          else if (accordionElement.querySelector(".financial-data-text")) type = "financial";
          else if (accordionElement.querySelector(".arxiv-paper-summary")) type = "arxiv"; // ADDED for ArXiv
          accordions.push({
            element: accordionElement,
            type: type,
          });
          accordionElement.remove();
        });

        // Clear the entire content div to remove any animated spans and streaming dots
        console.log(`[DEBUG] Clearing all content from response div before setting final content`);
        responseContentDiv.innerHTML = "";

        // Stop any ongoing stream animations by clearing the buffer and canceling animation frames
        console.log(`[DEBUG] Stopping animations for response ${listenerResponseCounter}`);
        streamDeltaBuffer = "";
        streamAnimationFrameRequested = false;
        streamFinalized = true; // Prevent any further modifications

        // Also clear any pending timeouts that might be adding more animated spans
        // (This is a safety measure since we can't easily track timeout IDs)
        console.log(`[DEBUG] Animation state cleared and stream finalized`);

        // Now set the final content
        try {
          responseContentDiv.innerHTML = safeRender(event.payload.full_content);
          console.log(`[DEBUG] Set final content for response ${listenerResponseCounter}`);
        } catch (e) {
          console.error("Error rendering markdown for main content:", e);
          responseContentDiv.textContent = event.payload.full_content;
        }

        // And again before inserting DOM elements
        if (responseContentDiv) {
          const order = ["article", "weather", "financial", "arxiv"]; // ADDED "arxiv" to order
          order.forEach((type) => {
            const accordionToPrepend = accordions.find((a) => a.type === type);
            if (accordionToPrepend && responseContentDiv) {
              responseContentDiv.insertBefore(
                accordionToPrepend.element,
                responseContentDiv.firstChild,
              );
            }
          });
        }
      }

      responseMessageDiv.classList.remove("streaming");

      // Handle reasoning accordion state at the end of the stream
      console.log("STREAM_END: responseMessageDiv:", responseMessageDiv);
      let reasoningAccordion = responseMessageDiv.querySelector(
        ".reasoning-accordion",
      ) as HTMLElement;
      console.log("STREAM_END: Found reasoningAccordion:", reasoningAccordion);
      if (reasoningAccordion) {
        const toggle = reasoningAccordion.querySelector(
          ".reasoning-accordion-toggle",
        ) as HTMLButtonElement;
        console.log(
          "STREAM_END: reasoningAccordion expanded BEFORE:",
          toggle?.getAttribute("aria-expanded"),
        );
      }

      if (event.payload.reasoning) {
        // If STREAM_END payload has reasoning, ensure accordion exists and update it
        if (!reasoningAccordion) {
          // Create new reasoning accordion if it doesn't exist
          const { accordion, content } = createCustomAccordion("Reasoning", "reasoning");
          const reasoningDiv = document.createElement("div");
          content.appendChild(reasoningDiv);
          reasoningAccordion = accordion;
          responseMessageDiv.appendChild(reasoningAccordion);
        }

        // Update existing or newly created reasoning accordion with final content
        const reasoningContent = reasoningAccordion.querySelector(".reasoning-content div");
        if (reasoningContent) {
          reasoningContent.setAttribute("data-raw-content", event.payload.reasoning);
          reasoningContent.innerHTML = safeRender(event.payload.reasoning);
        }
      }

      // If a reasoning accordion exists (either pre-existing or created above),
      // close it and set its toggle text.
      if (reasoningAccordion) {
        const toggle = reasoningAccordion.querySelector(
          ".reasoning-accordion-toggle",
        ) as HTMLButtonElement;
        const content = reasoningAccordion.querySelector(".reasoning-content") as HTMLElement;
        if (toggle && content) {
          console.log(
            "STREAM_END: Attempting to close reasoningAccordion. Current expanded state:",
            toggle.getAttribute("aria-expanded"),
          );
          toggle.setAttribute("aria-expanded", "false");
          content.classList.remove("open");
          console.log(
            "STREAM_END: reasoningAccordion expanded AFTER setting to false:",
            toggle.getAttribute("aria-expanded"),
          );
          toggle.textContent = "Show Reasoning";
          console.log("STREAM_END: Toggle text set to:", toggle.textContent);
        } else {
          console.log("STREAM_END: Toggle or content element not found for reasoningAccordion.");
        }
      } else {
        console.log("STREAM_END: No reasoningAccordion element found to close.");
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
    } else {
      // Fallback: if we can't find the specific response div, try to remove streaming dots from current global div
      console.log(
        `[DEBUG] No response divs found for counter ${listenerResponseCounter}, trying fallback cleanup`,
      );
      if (currentAssistantContentDiv) {
        const allFallbackDots = currentAssistantContentDiv.querySelectorAll(".streaming-dots");
        console.log(`[DEBUG] Found ${allFallbackDots.length} streaming dots in fallback cleanup`);
        allFallbackDots.forEach((dots, index) => {
          dots.remove();
          console.log(`[DEBUG] Removed fallback streaming dots ${index + 1}`);
        });
      }

      // Also try to clean up streaming dots from ALL response divs as a last resort
      console.log(`[DEBUG] Cleaning up streaming dots from all response divs`);
      responseDivMap.forEach((divs, counter) => {
        const allDots = divs.contentDiv.querySelectorAll(".streaming-dots");
        if (allDots.length > 0) {
          console.log(`[DEBUG] Found ${allDots.length} streaming dots in response ${counter}`);
          allDots.forEach((dots, index) => {
            dots.remove();
            console.log(`[DEBUG] Removed streaming dots ${index + 1} from response ${counter}`);
          });
        }
      });
    }
    if (messageInput) {
      messageInput.contentEditable = "true"; // MODIFIED
      // messageInput.focus();
    }

    // Only clear global state if this is the current active response
    if (listenerResponseCounter === responseCounter) {
      currentAssistantMessageDiv = null;
      currentAssistantContentDiv = null;
      isAIResponding = false; // Reset flag as AI has finished responding
    }

    if (chatHistory) chatHistory.scrollTop = chatHistory.scrollHeight;
  });

  if (unlistenStreamError) unlistenStreamError();
  unlistenStreamError = await listen<StreamErrorPayload>("STREAM_ERROR", (event) => {
    // Ignore events if this listener is not for the current response
    if (listenerResponseCounter !== responseCounter) {
      return;
    }

    console.error("STREAM_ERROR received:", event.payload);
    const responseDivs = responseDivMap.get(listenerResponseCounter);
    const responseMessageDiv = responseDivs?.messageDiv;
    const responseContentDiv = responseDivs?.contentDiv;

    if (responseMessageDiv && responseContentDiv) {
      // Remove streaming dots before setting error content
      const existingDots = responseContentDiv.querySelector(".streaming-dots");
      if (existingDots) {
        existingDots.remove();
      }
      responseContentDiv.innerHTML = safeRender(`Error: ${event.payload.error}`);
      responseMessageDiv.classList.remove("streaming");
      responseMessageDiv.classList.add("error"); // Optional: add error class for styling
    } else {
      // If no placeholder, add a new message for the error
      addMessageToHistory("Shard", `Error: ${event.payload.error}`);
    }
    if (messageInput) {
      messageInput.contentEditable = "true"; // MODIFIED
      // messageInput.focus();
    }

    // Only clear global state if this is the current active response
    if (listenerResponseCounter === responseCounter) {
      currentAssistantMessageDiv = null;
      currentAssistantContentDiv = null;
      isAIResponding = false; // Reset flag on stream error
    }
  });

  // --- WEATHER LOOKUP LISTENERS ---
  if (unlistenWeatherLookupStarted) unlistenWeatherLookupStarted();
  unlistenWeatherLookupStarted = await listen<WeatherLookupStartedPayload>(
    "WEATHER_LOOKUP_STARTED",
    (event) => {
      console.log("WEATHER_LOOKUP_STARTED received:", event.payload);
      const currentResponseContentDiv = responseDivMap.get(responseCounter)?.contentDiv;
      if (currentResponseContentDiv) {
        const existingStatus = currentResponseContentDiv.querySelector(
          ".weather-lookup-status-container",
        );
        if (existingStatus) existingStatus.remove();

        const existingDots = currentAssistantContentDiv?.querySelector(".streaming-dots");
        if (
          currentAssistantContentDiv &&
          currentAssistantContentDiv.querySelector(".streaming-dots") &&
          currentAssistantContentDiv.children.length === 1 &&
          currentAssistantContentDiv.firstChild === existingDots
        ) {
          const existingDots = currentAssistantContentDiv.querySelector(".streaming-dots");
          existingDots?.remove();
        }

        const lookupStatusDiv = document.createElement("div");
        lookupStatusDiv.classList.add("weather-lookup-status-container");

        const weatherIcon = createThermometerIcon();
        weatherIcon.classList.add("spinning-icon"); // Apply the generic spinning class
        lookupStatusDiv.appendChild(weatherIcon);

        const statusText = document.createElement("span");
        statusText.textContent = `Fetching weather for: "${event.payload.location}"...`;
        statusText.classList.add("weather-lookup-status-text");
        lookupStatusDiv.appendChild(statusText);

        if (currentAssistantContentDiv) {
          currentAssistantContentDiv.insertBefore(
            lookupStatusDiv,
            currentAssistantContentDiv.firstChild,
          );
        }
        if (chatHistory) chatHistory.scrollTop = chatHistory.scrollHeight;
      }
    },
  );

  if (unlistenWeatherLookupCompleted) unlistenWeatherLookupCompleted();
  unlistenWeatherLookupCompleted = await listen<WeatherLookupCompletedPayload>(
    "WEATHER_LOOKUP_COMPLETED",
    (event) => {
      console.log("WEATHER_LOOKUP_COMPLETED received:", event.payload);
      const currentResponseContentDiv = responseDivMap.get(responseCounter)?.contentDiv;
      if (currentResponseContentDiv) {
        const statusContainer = currentResponseContentDiv.querySelector(
          ".weather-lookup-status-container",
        );
        if (statusContainer) statusContainer.remove();

        const { accordion, toggle, content } = createCustomAccordion(
          `Weather Information for: "${event.payload.location}"`,
          "web-search",
        );

        const weatherIcon = createThermometerIcon(); // Static icon for summary
        addIconToToggle(toggle, weatherIcon);

        const weatherContentDiv = content;
        weatherContentDiv.classList.add("weather-content");

        if (
          event.payload.success &&
          event.payload.temperature !== null &&
          event.payload.temperature !== undefined &&
          event.payload.location
        ) {
          // Success with data
          const weatherText = document.createElement("p");
          weatherText.classList.add("weather-info-text");
          let displayText = `Temperature: ${event.payload.temperature.toFixed(1)}°${event.payload.unit || "C"}`;
          if (event.payload.description) {
            displayText += `\nDescription: ${event.payload.description}`;
          }
          weatherText.style.whiteSpace = "pre-wrap";
          weatherText.textContent = displayText;
          weatherContentDiv.appendChild(weatherText);
        } else {
          // Error or No Data case
          const errorParagraph = document.createElement("p");
          errorParagraph.classList.add("weather-lookup-error-text"); // For styling if needed

          if (event.payload.error) {
            console.error("Weather lookup failed:", event.payload.error);
            errorParagraph.textContent = `Weather lookup for "${event.payload.location}" failed: ${event.payload.error}`;
          } else if (
            event.payload.success &&
            (event.payload.temperature === null || event.payload.temperature === undefined) &&
            event.payload.location
          ) {
            // Success, but no specific temperature data
            errorParagraph.textContent = `Could not retrieve weather data for "${event.payload.location}".`;
          } else {
            // General fallback, should ideally not be reached if payload structure is consistent
            errorParagraph.textContent = `An unexpected issue occurred while fetching weather for "${event.payload.location}".`;
          }
          weatherContentDiv.appendChild(errorParagraph);

          const tipParagraph = document.createElement("p");
          tipParagraph.classList.add("weather-lookup-tip"); // For styling if needed
          tipParagraph.textContent = "If the place isn't found, try your zip code!";
          weatherContentDiv.appendChild(tipParagraph);

          // Optionally open the accordion if there's an error
          toggle.setAttribute("aria-expanded", "true");
          content.classList.add("open");
        }

        if (currentAssistantContentDiv) {
          currentAssistantContentDiv.insertBefore(accordion, currentAssistantContentDiv.firstChild);
        }

        // Ensure streaming dots are present if no other content is being streamed by the LLM yet
        if (
          currentAssistantContentDiv &&
          !currentAssistantContentDiv.querySelector(".streaming-dots") &&
          (currentAssistantContentDiv.innerHTML === "" || // Empty
            (currentAssistantContentDiv.children.length > 0 && // Only has accordions/tool messages
              currentAssistantContentDiv.querySelectorAll(
                ":not(.streaming-dots):not(.web-search-accordion):not(.tool-error-message):not(.tool-info-message)",
              ).length === 0))
        ) {
          currentAssistantContentDiv.appendChild(getStreamingDots());
        }
        if (chatHistory) chatHistory.scrollTop = chatHistory.scrollHeight;
      }
    },
  );
  // --- END WEATHER LOOKUP LISTENERS ---

  // --- ARXIV LOOKUP LISTENERS ---
  if (unlistenArxivLookupStarted) unlistenArxivLookupStarted();
  unlistenArxivLookupStarted = await listen<ArxivLookupStartedPayload>(
    "ARXIV_LOOKUP_STARTED",
    (event) => {
      console.log("ARXIV_LOOKUP_STARTED received:", event.payload);
      if (currentAssistantContentDiv) {
        const existingStatus = currentAssistantContentDiv.querySelector(
          ".arxiv-lookup-status-container",
        );
        if (existingStatus) existingStatus.remove();

        const existingDots = currentAssistantContentDiv.querySelector(".streaming-dots");
        if (
          existingDots &&
          currentAssistantContentDiv.children.length === 1 &&
          currentAssistantContentDiv.firstChild === existingDots
        ) {
          existingDots.remove();
        }

        const lookupStatusDiv = document.createElement("div");
        lookupStatusDiv.classList.add("arxiv-lookup-status-container"); // New class

        const arxivIconElement = createArxivIcon();
        arxivIconElement.classList.add("spinning-icon");
        lookupStatusDiv.appendChild(arxivIconElement);

        const statusText = document.createElement("span");
        statusText.textContent = `Fetching papers from ArXiv for: "${event.payload.query}"...`;
        statusText.classList.add("arxiv-lookup-status-text"); // New class
        lookupStatusDiv.appendChild(statusText);

        currentAssistantContentDiv.insertBefore(
          lookupStatusDiv,
          currentAssistantContentDiv.firstChild,
        );
        if (chatHistory) chatHistory.scrollTop = chatHistory.scrollHeight;
      }
    },
  );

  if (unlistenArxivLookupCompleted) unlistenArxivLookupCompleted();
  unlistenArxivLookupCompleted = await listen<ArxivLookupCompletedPayload>(
    "ARXIV_LOOKUP_COMPLETED",
    (event) => {
      console.log("ARXIV_LOOKUP_COMPLETED received:", event.payload);
      if (currentAssistantContentDiv) {
        const statusContainer = currentAssistantContentDiv.querySelector(
          ".arxiv-lookup-status-container",
        );
        if (statusContainer) statusContainer.remove();

        const { accordion, toggle, content } = createCustomAccordion(
          `ArXiv Results for: "${event.payload.query}"`,
          "arxiv-search", // New type for accordion
        );

        const arxivIconElement = createArxivIcon(); // Static icon
        addIconToToggle(toggle, arxivIconElement);

        const arxivContentDiv = content; // content is already the div we need
        arxivContentDiv.classList.add("arxiv-content"); // Add specific class if needed for styling

        if (event.payload.success && event.payload.results && event.payload.results.length > 0) {
          event.payload.results.forEach((paper) => {
            const paperDiv = document.createElement("div");
            paperDiv.classList.add("arxiv-paper-summary"); // Class for individual paper

            const titleEl = document.createElement("h4");
            titleEl.textContent = paper.title;
            paperDiv.appendChild(titleEl);

            const authorsEl = document.createElement("p");
            authorsEl.innerHTML = `<strong>Authors:</strong> ${paper.authors.join(", ")}`;
            paperDiv.appendChild(authorsEl);

            if (paper.published_date) {
              const publishedEl = document.createElement("p");
              publishedEl.innerHTML = `<strong>Published:</strong> ${paper.published_date}`;
              paperDiv.appendChild(publishedEl);
            }

            const summaryEl = document.createElement("p");
            summaryEl.classList.add("arxiv-summary-text");
            // Truncate summary if it's too long, or provide a "show more" later
            const summaryToShow = paper.summary;
            summaryEl.textContent = summaryToShow;
            paperDiv.appendChild(summaryEl);

            const pdfLinkEl = document.createElement("a");
            pdfLinkEl.href = paper.pdf_url;
            pdfLinkEl.textContent = "View PDF on ArXiv";
            pdfLinkEl.target = "_blank";
            pdfLinkEl.rel = "noopener noreferrer";
            pdfLinkEl.classList.add("arxiv-pdf-link");
            paperDiv.appendChild(pdfLinkEl);

            arxivContentDiv.appendChild(paperDiv);
          });
        } else {
          const messageParagraph = document.createElement("p");
          if (event.payload.error) {
            console.error("ArXiv lookup failed:", event.payload.error);
            messageParagraph.textContent = `ArXiv lookup for "${event.payload.query}" failed: ${event.payload.error}`;
          } else {
            messageParagraph.textContent = `No ArXiv papers found for "${event.payload.query}".`;
          }
          arxivContentDiv.appendChild(messageParagraph);
          // Optionally open the accordion if there's an error or no results
          toggle.setAttribute("aria-expanded", "true");
          content.classList.add("open");
        }

        currentAssistantContentDiv.insertBefore(accordion, currentAssistantContentDiv.firstChild);

        if (
          !currentAssistantContentDiv.querySelector(".streaming-dots") &&
          (currentAssistantContentDiv.innerHTML === "" ||
            (currentAssistantContentDiv.children.length > 0 &&
              currentAssistantContentDiv.querySelectorAll(
                ":not(.streaming-dots):not(.web-search-accordion):not(.tool-error-message):not(.tool-info-message)",
              ).length === 0))
        ) {
          currentAssistantContentDiv.appendChild(getStreamingDots());
        }
        if (chatHistory) chatHistory.scrollTop = chatHistory.scrollHeight;
      }
    },
  );
  // --- END ARXIV LOOKUP LISTENERS ---
}

// --- Function to add CSS styles for tool status containers ---
function addToolStatusStyles() {
  const style = document.createElement("style");
  style.textContent = `
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    .spinning-icon {
      animation: spin 2s linear infinite;
    }

    .article-lookup-status-container,
    .weather-lookup-status-container,
    .financial-data-status-container {
      background-color: transparent;
      border-radius: 8px;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      padding-left: 4px;
      gap: 8px;
      color: var(--text-color, #e0e0e0);
    }

    .article-lookup-status-text,
    .weather-lookup-status-text,
    .financial-data-status-text {
      font-size: 0.9em;
    }
  `;
  document.head.appendChild(style);
}

window.addEventListener("DOMContentLoaded", async () => {
  loadInitialSettings();
  setupStreamListeners(); // ADDED: Setup listeners on DOM load
  addToolStatusStyles(); // ADDED: Call the function to inject styles
  await setInitialWindowGeometry(); // Set fixed window size and position
  updateInputAreaLayout(); // ADDED: Initial layout setup
  updatePlaceholderState(); // ADDED: Initial placeholder check

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
  document.addEventListener("mousedown", async (event) => {
    const target = event.target as HTMLElement;
    // Basic check to avoid errors if target is not an HTMLElement (e.g., SVGElement in some cases, though less common for this specific problem)
    if (!target || typeof target.closest !== "function") {
      console.log(
        "[ClickThrough] Event target is not an HTMLElement or doesn't support 'closest'. Ignoring.",
      );
      return;
    }
    console.log(
      `[ClickThrough] Mousedown event. Target: <${target.tagName}> id='${target.id || "none"}' class='${target.className || "none"}'`,
    );

    // Define selectors for all elements that should remain interactive
    const interactiveSelectors = [
      "#message-input",
      "#input-image-preview",
      "#ocr-icon-container",
      "#clear-chat-button",
      "#settings-toggle",
      "#settings-panel", // settings-panel and all its children
      "#chat-history > *", // Any direct child of chat-history (messages, accordions, etc.)
      "#input-area",
      // General HTML tags that are usually interactive by nature
      "button",
      "textarea",
      "input",
      "select",
      "details",
      "summary",
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
      console.log(
        `[ClickThrough] Click target matched interactive selector: '${matchedSelector}'. Window remains interactive.`,
      );
      // Ensure the window is interactive if an interactive element is clicked.
      try {
        await appWindow.setIgnoreCursorEvents(false);
        // console.log("[ClickThrough] Ensured cursor events are enabled due to interactive click.");
      } catch (error) {
        console.error(
          "[ClickThrough] Error ensuring cursor events enabled on interactive click:",
          error,
        );
      }
    } else {
      console.log(
        "[ClickLogic] Click target did not match interactive selectors. Emitting 'js-request-toggle-window' to backend.",
      );
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
      }, 500);
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
    // Set initial height correctly - not strictly needed for div, but ensure no fixed style
    // messageInput.style.height = initialTextareaHeight; // REMOVED for div
    // messageInput.style.overflowY = "hidden"; // REMOVED for div, now 'auto' via CSS

    // MODIFIED: Event listeners for contenteditable div
    messageInput.addEventListener("input", () => {
      updatePlaceholderState(); // ADDED: Update placeholder on any input
      updateInputAreaLayout();
    });

    messageInput.addEventListener("keydown", async (event) => {
      // Changed to keydown for Backspace
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        await handleSendMessage();
      } else if (event.key === "Backspace") {
        const textContent = getTextFromContentEditable(messageInput);
        // Check if the div is visually empty (only contains our image wrapper or is truly empty)
        const firstChild = messageInput.firstChild;
        const isEffectivelyEmpty =
          !textContent ||
          (firstChild &&
            (firstChild as HTMLElement).classList?.contains("inline-image-wrapper") &&
            !firstChild.nextSibling &&
            messageInput.textContent?.trim() === "");

        if (isEffectivelyEmpty && currentImageBase64) {
          event.preventDefault();
          await clearInlineImageAndData();
        }
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
    clearChatButton.addEventListener("click", handleClearOrUndoChat);
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

  // ADDED: Listen for window resize to update layout
  window.addEventListener("resize", updateInputAreaLayout);
});

// --- Event Listener for Window Toggle ---
const FADE_DURATION = 300; // ms - Should match CSS transition duration
const FADE_DURATION_CHATHISTORY = 200; // ms - Should match CSS transition duration

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

// Custom accordion helper functions
function createCustomAccordion(
  title: string,
  type: "reasoning" | "web-search" | "arxiv-search" | "ocr" | "image-preview", // ADDED "image-preview"
): { accordion: HTMLElement; toggle: HTMLButtonElement; content: HTMLElement } {
  const accordion = document.createElement("div");
  if (type === "reasoning") {
    accordion.className = "reasoning-accordion";
  } else if (type === "ocr") {
    accordion.className = "ocr-accordion";
  } else if (type === "image-preview") {
    // New type
    accordion.className = "image-preview-accordion";
  } else {
    accordion.className = "web-search-accordion"; // Keep general class for web-search style
  }

  const toggle = document.createElement("button");
  if (type === "reasoning") {
    toggle.className = "reasoning-accordion-toggle";
  } else if (type === "ocr") {
    toggle.className = "ocr-accordion-toggle";
  } else if (type === "image-preview") {
    // New type
    toggle.className = "image-preview-accordion-toggle";
  } else {
    toggle.className = "web-search-accordion-toggle";
  }
  toggle.setAttribute("aria-expanded", "false");
  toggle.setAttribute("aria-controls", `${type}-content-${Date.now()}`); // Use type in ID
  toggle.textContent = title;

  const content = document.createElement("div");
  if (type === "reasoning") {
    content.className = "reasoning-content";
  } else if (type === "ocr") {
    content.className = "ocr-content";
  } else if (type === "image-preview") {
    // New type
    content.className = "image-preview-content"; // Div to hold the image preview
  } else {
    content.className = "web-search-content"; // Keep general class
  }
  content.id = toggle.getAttribute("aria-controls")!;
  content.setAttribute("role", "region");
  content.setAttribute("aria-labelledby", toggle.id || "");

  // Add click handler
  toggle.addEventListener("click", () => {
    const isOpen = toggle.getAttribute("aria-expanded") === "true";
    toggle.setAttribute("aria-expanded", (!isOpen).toString());
    content.classList.toggle("open", !isOpen);
  });

  accordion.appendChild(toggle);
  accordion.appendChild(content);

  return { accordion, toggle, content };
}

function addIconToToggle(toggle: HTMLButtonElement, icon: SVGElement) {
  toggle.insertBefore(icon, toggle.firstChild);
  toggle.insertBefore(document.createTextNode(" "), toggle.firstChild?.nextSibling || null);
}
