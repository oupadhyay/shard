import './style.css'
import { core } from '@tauri-apps/api'
import { invoke } from '@tauri-apps/api/core'

// DOM Elements
const apiKeyInput = document.getElementById('api-key-input') as HTMLInputElement
const saveApiKeyButton = document.getElementById('save-api-key') as HTMLButtonElement
const settingsStatus = document.getElementById('settings-status') as HTMLParagraphElement
const settingsToggle = document.getElementById('settings-toggle') as HTMLButtonElement
const settingsPanel = document.getElementById('settings-panel') as HTMLDivElement
const clearChatButton = document.getElementById('clear-chat-button') as HTMLButtonElement
const modelSelect = document.getElementById('model-select') as HTMLSelectElement

const messageInput = document.getElementById('message-input') as HTMLTextAreaElement
const inputImagePreview = document.getElementById('input-image-preview') as HTMLImageElement
const chatHistory = document.getElementById('chat-history') as HTMLDivElement
const ocrIconContainer = document.getElementById('ocr-icon-container') as HTMLDivElement
const statusMessage = document.getElementById('status-message') as HTMLParagraphElement

// Model mapping: Display Name -> OpenRouter Identifier
const modelMap: { [key: string]: string } = {
  "Deepseek R1 (free)": "deepseek/deepseek-r1:free",
  "Deepseek V3 (free)": "deepseek/deepseek-chat-v3-0324:free",
  "Gemini 2.0 Flash (free)": "google/gemini-2.0-flash-exp:free",
};

// Define the expected response structure from the backend
interface ModelResponse {
  content: string;
  reasoning: string | null; // Reasoning can be null
}

// Define the structure returned by the capture command
interface CaptureResult {
    ocr_text: string;
    image_base64: string | null;
    temp_path: string | null;
}

// State Variables
let isChatCleared = false; // Flag to track if chat was cleared
let currentOcrText: string | null = null;
let currentImageBase64: string | null = null;
let currentTempScreenshotPath: string | null = null;

// --- Populate Model Select Dropdown ---
function populateModelSelect() {
  if (!modelSelect) return;
  modelSelect.innerHTML = ''; // Clear existing options
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
    const key = await invoke<string>('get_api_key')
    apiKeyInput.value = key || ''
    if (key) {
      console.log('API Key loaded.')
    } else {
      console.log('API Key not set.')
      // settingsStatus.textContent = 'API Key not set.'; // Optional: notify user
    }
  } catch (error) {
    console.error('Failed to load API key:', error)
    settingsStatus.textContent = `Error loading key: ${error}`
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
          console.warn(`Saved model ID "${selectedModelId}" not found in dropdown options. Defaulting to first option.`);
           // Optionally set to a default if the saved one is invalid or not found
           // modelSelect.selectedIndex = 0; // Or handle as needed
        }
      } else {
        console.log("No selected model ID returned or modelSelect.options not available. Using default selection.");
      }
    } catch (error) {
      console.error("Failed to load selected model:", error);
      if(settingsStatus) settingsStatus.textContent = `Error loading model: ${error}`;
    }
  }
}

// --- API Key Management ---
// async function loadApiKey() {
//   try {
//     const key = await invoke<string>('get_api_key')
//     apiKeyInput.value = key || ''
//     if (key) {
//       console.log('API Key loaded.')
//     } else {
//       console.log('API Key not set.')
//     }
//   } catch (error) {
//     console.error('Failed to load API key:', error)
//     settingsStatus.textContent = `Error loading key: ${error}`
//   }
// }

async function saveApiKey() {
  const key = apiKeyInput.value.trim()
  if (!key) {
    settingsStatus.textContent = 'API Key cannot be empty.'
    return
  }
  try {
    await core.invoke('set_api_key', { key })
    settingsStatus.textContent = 'API Key saved!'
    setTimeout(() => settingsStatus.textContent = '', 3000) // Clear status after 3s
  } catch (error) {
    console.error('Failed to save API key:', error)
    settingsStatus.textContent = 'Error saving API key.'
  }
}

// --- Helper to auto-resize textarea ---
const initialTextareaHeight = 'calc(2em * 1.4)'; // Store initial height
function autoResizeTextarea() {
    if (!messageInput) return;
    // Temporarily shrink height to get accurate scrollHeight
    messageInput.style.height = 'auto';
    // Set height based on content, but don't exceed a max
    const newHeight = Math.min(messageInput.scrollHeight, 200); // Limit max height to 200px (adjust as needed)
    messageInput.style.height = `${newHeight}px`;
    // Show scrollbar if content exceeds max height
    messageInput.style.overflowY = newHeight >= 200 ? 'auto' : 'hidden';
}

// --- Chat Functionality ---
function addMessageToHistory(sender: 'You' | 'Shard', content: string, reasoning: string | null = null) {
  const messageDiv = document.createElement('div')
  messageDiv.classList.add('message')
  messageDiv.classList.add(sender === 'You' ? 'user' : 'assistant')

  const senderStrong = document.createElement('strong')
  senderStrong.textContent = sender
  messageDiv.appendChild(senderStrong)

  // Add main content
  // Use innerHTML to render potential markdown/formatting later if needed, for now textNode is safer
  // const contentNode = document.createTextNode(content);
  // messageDiv.appendChild(contentNode);
  const contentDiv = document.createElement('div');
  contentDiv.textContent = content; // Or .innerHTML = marked(content) if using a markdown parser
  messageDiv.appendChild(contentDiv);


  // Add reasoning accordion if reasoning is present for assistant messages
  if (sender === 'Shard' && reasoning) {
    const details = document.createElement('details');
    details.classList.add('reasoning-accordion');

    const summary = document.createElement('summary');
    summary.textContent = 'Show Reasoning';
    details.appendChild(summary);

    const reasoningContent = document.createElement('div');
    reasoningContent.classList.add('reasoning-content');
    // Display reasoning as preformatted text for now
    const pre = document.createElement('pre');
    pre.textContent = reasoning;
    reasoningContent.appendChild(pre);

    details.appendChild(reasoningContent);
    messageDiv.appendChild(details);
  }


  chatHistory.appendChild(messageDiv)
  chatHistory.scrollTop = chatHistory.scrollHeight // Auto-scroll to bottom
}

// --- Helper to update Input Preview and Tooltip ---
function updateInputAreaForCapture() {
    if (currentOcrText) {
        messageInput.title = currentOcrText; // Set tooltip on input
    } else {
        messageInput.title = ''; // Clear tooltip
    }

    if (currentImageBase64) {
        inputImagePreview.src = `data:image/png;base64,${currentImageBase64}`;
        inputImagePreview.classList.remove('hidden');
    } else {
        inputImagePreview.src = '';
        inputImagePreview.classList.add('hidden');
    }
}

// --- Clear Chat Handler ---
function clearChatHistory() {
    if (chatHistory) chatHistory.innerHTML = '';
    isChatCleared = true; // Set flag when clearing
    console.log('Chat history cleared.');
    // Hide clear button again
    if (clearChatButton) clearChatButton.classList.add('hidden');

    // Clear capture state and cleanup temp file if necessary
    if (currentTempScreenshotPath) {
        console.log("Cleanup requested for temp screenshot:", currentTempScreenshotPath);
        invoke('cleanup_temp_screenshot', { path: currentTempScreenshotPath })
            .then(() => console.log("Temp screenshot cleanup successful."))
            .catch(err => console.error("Error cleaning up temp screenshot:", err));
    }
    currentOcrText = null;
    currentImageBase64 = null;
    currentTempScreenshotPath = null;
    updateInputAreaForCapture(); // Clear preview and tooltip

    if (statusMessage) statusMessage.textContent = ''; // Clear status
}

// --- Capture OCR Handler ---
async function handleCaptureOcr() {
    console.log('Capture OCR initiated');
    isChatCleared = false; // Allow new messages
    if (statusMessage) {
        statusMessage.textContent = 'Starting screen capture...';
        statusMessage.style.display = 'block';
    }
    // Clear previous capture state *before* starting new capture
    if (currentTempScreenshotPath) {
        console.log("Cleaning up previous temp screenshot:", currentTempScreenshotPath);
        await invoke('cleanup_temp_screenshot', { path: currentTempScreenshotPath })
            .catch(err => console.error("Error cleaning up previous temp screenshot:", err));
        currentTempScreenshotPath = null; // Ensure path is cleared even if cleanup fails
    }
    currentOcrText = null;
    currentImageBase64 = null;
    updateInputAreaForCapture();

    // Visually indicate loading
    if (ocrIconContainer) ocrIconContainer.style.opacity = '0.5';

    try {
        const result = await core.invoke<CaptureResult>('capture_interactive_and_ocr');
        console.log('Capture Result:', result);

        currentOcrText = result.ocr_text;
        currentImageBase64 = result.image_base64;
        currentTempScreenshotPath = result.temp_path;

        updateInputAreaForCapture(); // Update input tooltip and image preview

        if (statusMessage) {
            if (currentOcrText || currentImageBase64) {
                statusMessage.textContent = 'Capture complete. OCR text added as tooltip to input.';
                 // Auto-hide status after a delay
                 setTimeout(() => {
                    if (statusMessage && statusMessage.textContent === 'Capture complete. OCR text added as tooltip to input.') {
                      statusMessage.style.display = 'none';
                      statusMessage.textContent = '';
                    }
                  }, 4000);
            } else {
                statusMessage.textContent = 'Capture complete, but no image or text was processed.';
            }
        }

    } catch (error) {
        console.error('Error during interactive capture/OCR:', error);
        const errorMessage = typeof error === 'string' ? error : 'Capture cancelled or failed.';
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
                  statusMessage.style.display = 'none';
                  statusMessage.textContent = '';
                }
              }, 5000);
        }
    } finally {
        // Re-enable icon
         if (ocrIconContainer) ocrIconContainer.style.opacity = '1';
    }
}

// --- Send Message Handler ---
async function handleSendMessage() {
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

    addMessageToHistory('You', textToDisplay);
    messageInput.value = ''; // Clear input field now
    messageInput.disabled = true;
    if(messageInput.title) messageInput.title = ''; // Clear tooltip immediately
    messageInput.style.height = initialTextareaHeight; // Reset height
    messageInput.style.overflowY = 'hidden'; // Hide scrollbar again
    if(!inputImagePreview.classList.contains('hidden')) {
        inputImagePreview.classList.add('hidden'); // Hide preview immediately
        inputImagePreview.src = '';
    }

    // Show clear button if it was hidden
    if (clearChatButton?.classList.contains('hidden')) {
        clearChatButton.classList.remove('hidden');
    }

    // Show thinking indicator
    const thinkingDiv = document.createElement('div');
    thinkingDiv.classList.add('message', 'assistant', 'thinking');
    thinkingDiv.innerHTML = `<strong>Shard</strong><div class="dots-container"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>`;
    if (chatHistory) {
        chatHistory.appendChild(thinkingDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    const removeThinkingIndicator = () => {
        const thinkingMsg = chatHistory?.querySelector('.message.thinking');
        if (thinkingMsg) {
            chatHistory.removeChild(thinkingMsg);
        }
    };

    try {
        const response = await core.invoke<ModelResponse>('send_text_to_model', { text: textToSend });
        removeThinkingIndicator();

        // Check if chat was cleared while waiting for response
        if (isChatCleared) {
            console.log("Chat was cleared, discarding incoming model response.");
            isChatCleared = false; // Reset flag
            messageInput.disabled = false;
            messageInput.focus();
            return; // Don't add the message
        }

        addMessageToHistory('Shard', response.content, response.reasoning);

        // Cleanup temp file ONLY after successful send
        if (tempPathToClean) {
            console.log("Cleaning up temp screenshot after successful send:", tempPathToClean);
            invoke('cleanup_temp_screenshot', { path: tempPathToClean })
                .catch(err => console.error("Error cleaning up temp screenshot post-send:", err));
        }

    } catch (error) {
         removeThinkingIndicator();
        let errorMessage = 'An error occurred.';
        if (typeof error === 'string') {
            errorMessage = error;
        } else if (error instanceof Error) {
            errorMessage = error.message;
        }
        console.error('Failed to send message:', error);
        addMessageToHistory('Shard', `Error: ${errorMessage}`);

        // Even on error, if we *tried* to send OCR text, attempt cleanup
        if (tempPathToClean) {
            console.warn("Cleaning up temp screenshot after failed send:", tempPathToClean);
            invoke('cleanup_temp_screenshot', { path: tempPathToClean })
                .catch(err => console.error("Error cleaning up temp screenshot post-failure:", err));
        }
    } finally {
        // Re-enable input regardless of success/failure
        if (messageInput) {
            messageInput.disabled = false;
            messageInput.focus();
        }
        // Ensure preview/tooltip are cleared in case they weren't before
        updateInputAreaForCapture();
    }
}

// --- Event Listeners ---
window.addEventListener('DOMContentLoaded', () => {
  loadInitialSettings();

  if (saveApiKeyButton) {
    saveApiKeyButton.addEventListener('click', saveApiKey)
  }

  if (messageInput) {
    // Set initial height correctly
    messageInput.style.height = initialTextareaHeight;
    messageInput.style.overflowY = 'hidden';

    // Add input event listener for auto-resizing
    messageInput.addEventListener('input', autoResizeTextarea);

    messageInput.addEventListener('keypress', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) { // Allow Shift+Enter for newlines if desired
        event.preventDefault(); // Prevent default Enter behavior (like adding newline)
        handleSendMessage();
      }
    });
  }

  if (settingsToggle) {
    settingsToggle.addEventListener('click', () => {
      if (settingsPanel.style.display === 'none' || settingsPanel.style.display === '') {
        settingsPanel.style.display = 'block'
      } else {
        settingsPanel.style.display = 'none'
      }
    })
  }

  if (clearChatButton) {
    clearChatButton.addEventListener('click', clearChatHistory)
  }

  // Model selection change listener
  if (modelSelect) {
    modelSelect.addEventListener("change", async (event) => {
      const selectedModelId = (event.target as HTMLSelectElement).value;
      console.log("Model selection changed to:", selectedModelId);
      try {
        await invoke("set_selected_model", { modelName: selectedModelId });
        console.log("Successfully saved selected model:", selectedModelId);
        if (settingsStatus) {
            settingsStatus.textContent = "Model selection saved.";
            setTimeout(() => settingsStatus.textContent = '', 3000);
        }
      } catch (error) {
        console.error("Failed to save selected model:", error);
        if (settingsStatus) settingsStatus.textContent = `Error saving model: ${error}`;
      }
    });
  }

  // Add listener to the OCR icon container
  if (ocrIconContainer) {
    ocrIconContainer.addEventListener('click', handleCaptureOcr);
  }
})
