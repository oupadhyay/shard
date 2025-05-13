import './style.css'
import { core } from '@tauri-apps/api'
import { invoke } from '@tauri-apps/api/core'
import { listen } from '@tauri-apps/api/event'
import { Window } from '@tauri-apps/api/window'

// DOM Elements
const apiKeyInput = document.getElementById('api-key-input') as HTMLInputElement
const geminiApiKeyInput = document.getElementById('gemini-api-key-input') as HTMLInputElement
const settingsStatus = document.getElementById('settings-status') as HTMLParagraphElement
const apiKeyStatusIcon = document.getElementById('api-key-status-icon') as HTMLSpanElement
const settingsToggle = document.getElementById('settings-toggle') as HTMLButtonElement
const settingsPanel = document.getElementById('settings-panel') as HTMLDivElement
const clearChatButton = document.getElementById('clear-chat-button') as HTMLButtonElement
const modelSelect = document.getElementById('model-select') as HTMLSelectElement

const FADE_DURATION_SETTINGS = 80; // Duration for settings panel fade

// Create and add close button to settings panel
if (settingsPanel) {
  const closeButton = document.createElement('button');
  closeButton.id = 'settings-close';
  closeButton.innerHTML = '×'; // Using × character for the X
  closeButton.title = 'Close Settings';
  settingsPanel.appendChild(closeButton);

  // Add click handler for close button
  closeButton.addEventListener('click', () => {
    settingsPanel.classList.remove('fade-in-settings'); // Start fade-out
    setTimeout(() => {
      settingsPanel.style.display = 'none';
    }, FADE_DURATION_SETTINGS);
    document.removeEventListener('click', handleClickOutsideSettings); // Remove listener
  });
}

const messageInput = document.getElementById('message-input') as HTMLTextAreaElement
const inputImagePreview = document.getElementById('input-image-preview') as HTMLImageElement
const chatHistory = document.getElementById('chat-history') as HTMLDivElement
const ocrIconContainer = document.getElementById('ocr-icon-container') as HTMLDivElement
const statusMessage = document.getElementById('status-message') as HTMLParagraphElement

// Model mapping: Display Name -> OpenRouter Identifier
const modelMap: { [key: string]: string } = {
  "Deepseek R1 (free)": "deepseek/deepseek-r1:free",
  "Deepseek V3 (free)": "deepseek/deepseek-chat-v3-0324:free",
  "Gemini 2.0 Flash": "gemini-2.0-flash",
  "Gemini 2.5 Flash": "gemini-2.5-flash-preview-04-17",
  "Gemini 2.5 Pro": "gemini-2.5-pro-preview-05-06"
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
    if (apiKeyInput) apiKeyInput.value = key || ''
    if (key) {
      console.log('API Key loaded.')
    } else {
      console.log('API Key not set.')
      // settingsStatus.textContent = 'API Key not set.'; // Optional: notify user
    }
  } catch (error) {
    console.error('Failed to load API key:', error)
    if (settingsStatus) settingsStatus.textContent = `Error loading key: ${error}`
  }

  // Load Gemini API Key
  if (geminiApiKeyInput) {
    try {
      const geminiKey = await invoke<string>('get_gemini_api_key');
      geminiApiKeyInput.value = geminiKey || '';
      if (geminiKey) {
        console.log('Gemini API Key loaded.');
      } else {
        console.log('Gemini API Key not set.');
      }
    } catch (error) {
      console.error('Failed to load Gemini API key:', error);
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

  if (apiKeyInput) {
    // Add input event listener with debounce for auto-saving
    let saveTimeout: number;
    apiKeyInput.addEventListener('input', () => {
      clearTimeout(saveTimeout);
      const key = apiKeyInput.value.trim();

      // Update immediately if clearing the key
      if (!key) {
        core.invoke('set_api_key', { key });
        if (settingsStatus) settingsStatus.textContent = 'API Key cleared';
        if (apiKeyStatusIcon) apiKeyStatusIcon.classList.remove('visible');
        setTimeout(() => {
          if (settingsStatus && settingsStatus.textContent === 'API Key cleared') settingsStatus.textContent = '';
        }, 3000);
        return;
      }

      // Otherwise, debounce the save
      saveTimeout = setTimeout(async () => {
        try {
          await core.invoke('set_api_key', { key });
          if (settingsStatus) settingsStatus.textContent = '';
          if (apiKeyStatusIcon) {
            apiKeyStatusIcon.classList.add('visible');
            setTimeout(() => {
              apiKeyStatusIcon.classList.remove('visible');
            }, 2000);
          }
        } catch (error) {
          console.error('Failed to save API key:', error);
          if (apiKeyStatusIcon) apiKeyStatusIcon.classList.remove('visible');
          if (settingsStatus) settingsStatus.textContent = 'Error saving API key.';
        }
      }, 500); // Wait 500ms after last keystroke before saving
    });
  }

  // Listener for Gemini API Key input
  if (geminiApiKeyInput) {
    let geminiSaveTimeout: number;
    geminiApiKeyInput.addEventListener('input', () => {
      clearTimeout(geminiSaveTimeout);
      const key = geminiApiKeyInput.value.trim();

      if (!key) {
        core.invoke('set_gemini_api_key', { key });
        if (settingsStatus) settingsStatus.textContent = 'Gemini API Key cleared';
        if (apiKeyStatusIcon) apiKeyStatusIcon.classList.remove('visible');
        setTimeout(() => {
          if (settingsStatus && settingsStatus.textContent === 'Gemini API Key cleared') settingsStatus.textContent = '';
        }, 3000);
        return;
      }

      geminiSaveTimeout = setTimeout(async () => {
        try {
          await core.invoke('set_gemini_api_key', { key });
          if (settingsStatus) settingsStatus.textContent = '';
          if (apiKeyStatusIcon) {
            apiKeyStatusIcon.classList.add('visible');
            setTimeout(() => {
              apiKeyStatusIcon.classList.remove('visible');
            }, 2000);
          }
        } catch (error) {
          console.error('Failed to save Gemini API key:', error);
          if (apiKeyStatusIcon) apiKeyStatusIcon.classList.remove('visible');
          if (settingsStatus) settingsStatus.textContent = 'Error saving Gemini API key.';
        }
      }, 500);
    });
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
    settingsToggle.addEventListener('click', (_event) => {
      if (settingsPanel.style.display === 'none' || settingsPanel.style.display === '') {
        settingsPanel.style.display = 'block';
        // Force reflow before adding class to ensure transition happens
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            settingsPanel.classList.add('fade-in-settings');
          });
        });
        setTimeout(() => document.addEventListener('click', handleClickOutsideSettings), 0);
      } else {
        settingsPanel.classList.remove('fade-in-settings'); // Start fade-out
        setTimeout(() => {
          settingsPanel.style.display = 'none';
        }, FADE_DURATION_SETTINGS);
        document.removeEventListener('click', handleClickOutsideSettings);
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

// --- Event Listener for Window Toggle ---
const FADE_DURATION = 300; // ms - Should match CSS transition duration

// --- Function to handle clicks outside the settings panel ---
function handleClickOutsideSettings(event: MouseEvent) {
  if (settingsPanel && settingsPanel.style.display === 'block') {
    // Check if the click is outside the panel AND not on the toggle button itself
    if (!settingsPanel.contains(event.target as Node) && event.target !== settingsToggle) {
      settingsPanel.classList.remove('fade-in-settings'); // Start fade-out
      setTimeout(() => {
        settingsPanel.style.display = 'none';
      }, FADE_DURATION_SETTINGS);
      document.removeEventListener('click', handleClickOutsideSettings);
    }
  }
}

listen('toggle-main-window', async () => {
  console.log('toggle-main-window event received from backend!');
  const mainWindow = await Window.getByLabel('main');
  if (!mainWindow) {
    console.error("Main window not found by label 'main'!");
    return;
  }

  const isVisible = await mainWindow.isVisible();
  const bodyElement = document.body;

  if (isVisible) {
    console.log('Fading out window...');
    bodyElement.classList.add('fade-out');
    bodyElement.classList.remove('fade-in'); // Ensure fade-in is removed

    // Wait for fade-out animation to complete before hiding
    setTimeout(async () => {
      await mainWindow.hide();
      bodyElement.classList.remove('fade-out'); // Clean up class
      console.log('Window hidden.');
    }, FADE_DURATION);

  } else {
    console.log('Fading in window...');
    // Ensure opacity is 0 before showing if using fade-in class
    bodyElement.style.opacity = '0'; // Start transparent
    bodyElement.classList.remove('fade-out'); // Ensure fade-out is removed

    await mainWindow.show(); // Show the (transparent) window
    await mainWindow.setFocus(); // Focus it

    // Force reflow/repaint before adding fade-in class might be needed in some cases
    // but often just adding the class works.
    requestAnimationFrame(() => {
      bodyElement.style.opacity = ''; // Reset opacity for CSS transition
      bodyElement.classList.add('fade-in');
      console.log('Fade-in class added.');

      // Optional: Remove fade-in class after animation completes to reset state
      setTimeout(() => {
        bodyElement.classList.remove('fade-in');
      }, FADE_DURATION);
    });
  }
});

console.log('Frontend listener for toggle-main-window set up.');
