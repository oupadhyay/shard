use base64::{engine::general_purpose, Engine as _}; // Added base64 import
use image::{DynamicImage, ImageFormat};
use reqwest;
use serde::{Deserialize, Serialize};
use serde_json;
use std::env; // For temp_dir
use std::fs;
use std::io::Cursor;
use std::path::PathBuf;
use std::process::Command;
use tauri::PhysicalPosition;
use tauri::{AppHandle, Emitter, Manager, WindowEvent, Window}; // Added Emitter and Window
use tauri_plugin_global_shortcut::{
    self as tauri_gs, GlobalShortcutExt, Shortcut, ShortcutEvent, ShortcutState,
};
use tesseract; // Uncommented
use uuid::Uuid; // For unique filenames // Added for base64 encoding // Plugin imports

#[cfg(target_os = "windows")]
use arboard::Clipboard;
#[cfg(target_os = "windows")]
use std::thread;
#[cfg(target_os = "windows")]
use std::time::Duration;

// Default model if none is selected
const DEFAULT_MODEL: &str = "deepseek/deepseek-chat-v3-0324:free";

// --- Config Structures ---
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
struct AppConfig {
    api_key: Option<String>,
    selected_model: Option<String>,
    gemini_api_key: Option<String>, // Added for Gemini
}

const CONFIG_FILENAME: &str = "config.toml";

// Define the structure returned by the capture command
#[derive(Serialize, Deserialize, Debug, Clone)]
struct CaptureResult {
    ocr_text: String,
    image_base64: Option<String>, // Option<> in case image loading/encoding fails
    temp_path: Option<String>,    // Path to the temp file created by screencapture/clipboard save
}

// --- Config Helper Functions ---
fn get_config_path(app_handle: &AppHandle) -> Result<PathBuf, String> {
    let resolver = app_handle.path();
    match resolver.app_config_dir() {
        Ok(dir) => Ok(dir.join(CONFIG_FILENAME)),
        Err(e) => Err(format!("Failed to get app config directory: {}", e)),
    }
}

fn load_config(app_handle: &AppHandle) -> Result<AppConfig, String> {
    let config_path = get_config_path(app_handle)?;
    if !config_path.exists() {
        log::info!(
            "Config file not found at {:?}, returning default.",
            config_path
        );
        return Ok(AppConfig::default());
    }
    log::info!("Loading config from {:?}", config_path);
    let content = fs::read_to_string(&config_path)
        .map_err(|e| format!("Failed to read config file: {}", e))?;
    toml::from_str(&content).map_err(|e| format!("Failed to parse config file: {}", e))
}

fn save_config(app_handle: &AppHandle, config: &AppConfig) -> Result<(), String> {
    let config_path = get_config_path(app_handle)?;
    log::info!("Saving config to {:?}", config_path);
    if let Some(parent_dir) = config_path.parent() {
        if !parent_dir.exists() {
            fs::create_dir_all(parent_dir)
                .map_err(|e| format!("Failed to create config directory: {}", e))?;
            log::info!("Created config directory: {:?}", parent_dir);
        }
    }
    let toml_string =
        toml::to_string_pretty(config).map_err(|e| format!("Failed to serialize config: {}", e))?;
    fs::write(&config_path, toml_string).map_err(|e| format!("Failed to write config file: {}", e))
}

// Request Structures
#[derive(Deserialize, Serialize, Debug, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize, Debug)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: Option<bool>,
}

// Response Structures
// Define a structure to return both content and reasoning
#[derive(Serialize, Deserialize, Debug)]
struct ModelResponse {
    content: String,
    reasoning: Option<String>,
}

// --- Gemini API Structures ---
#[derive(Serialize, Deserialize, Debug)] // Deserialize needed for Candidate's content
struct GeminiPart {
    text: String,
}

#[derive(Serialize, Deserialize, Debug)] // Deserialize needed for Candidate's content
struct GeminiContent {
    parts: Vec<GeminiPart>,
    role: Option<String>, // Optional: "user" or "model"
}

// ADDED: Structures for GenerationConfig and ThinkingConfig for Gemini
#[derive(Serialize, Debug, Clone, Default)]
struct ThinkingConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    include_thoughts: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<i32>,
}

#[derive(Serialize, Debug, Default, Clone)]
struct GenerationConfigForGemini {
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<ThinkingConfig>,
    // In the future, other fields like temperature, maxOutputTokens can be added here
    // For example:
    // #[serde(skip_serializing_if = "Option::is_none")]
    // temperature: Option<f32>,
    // #[serde(skip_serializing_if = "Option::is_none")]
    // max_output_tokens: Option<i32>,
}

#[derive(Serialize, Debug)]
struct GeminiChatCompletionRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")] // ADDED
    generation_config: Option<GenerationConfigForGemini>, // ADDED
}

#[derive(Deserialize, Debug)]
struct GeminiCandidate {
    content: GeminiContent,
    // finish_reason: Option<String>,
    // safety_ratings: Option<Vec<serde_json::Value>>,
}

#[derive(Deserialize, Debug)]
struct GeminiChatCompletionResponse {
    candidates: Vec<GeminiCandidate>,
    // prompt_feedback: Option<serde_json::Value>,
}

// Structures for streaming OpenRouter events (OpenAI compatible)
#[derive(Serialize, Deserialize, Debug, Clone)] // Clone for emitting
struct StreamChoiceDelta {
    content: Option<String>, // Content is optional as some chunks might not have it
    role: Option<String>, // Role might appear in first chunk
}

#[derive(Serialize, Deserialize, Debug, Clone)] // Clone for emitting
struct StreamChoice {
    delta: StreamChoiceDelta,
    finish_reason: Option<String>,
    index: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)] // Clone for emitting
struct StreamingChatCompletionResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<StreamChoice>,
}

#[derive(Serialize, Clone)] // ADDED - Payload for STREAM_CHUNK event
struct StreamChunkPayload {
    delta: Option<String>,
}

#[derive(Serialize, Clone)] // ADDED - Payload for STREAM_END event
struct StreamEndPayload {
    full_content: String,
    reasoning: Option<String>, // Or whatever final data you want to send
}

#[derive(Serialize, Clone)] // ADDED - Payload for STREAM_ERROR event
struct StreamErrorPayload {
    error: String,
}

// --- Screen Capture & OCR Helper Functions ---
// Uncommented and kept as is
fn ocr_image_buffer(app_handle: &AppHandle, img_buffer: &DynamicImage) -> Result<String, String> {
    log::info!("Starting OCR process with tesseract crate for an image buffer.");
    let temp_dir_result = app_handle.path().app_cache_dir();
    let temp_dir = temp_dir_result.map_err(|e| {
        log::error!("Failed to get app cache directory: {}", e);
        format!("Failed to get app cache directory: {}", e)
    })?;
    if !temp_dir.exists() {
        fs::create_dir_all(&temp_dir)
            .map_err(|e| format!("Failed to create cache directory: {}", e))?;
    }
    let temp_filename = format!("{}.png", Uuid::new_v4().to_string());
    let temp_file_path = temp_dir.join(&temp_filename);

    log::info!(
        "Saving image to temporary file for OCR: {:?}",
        temp_file_path
    );
    img_buffer
        .save_with_format(&temp_file_path, ImageFormat::Png)
        .map_err(|e| {
            log::error!("Failed to save image to temp file: {}", e);
            format!("Failed to save image to temp file: {}", e)
        })?;

    let ocr_text_result = || -> Result<String, String> {
        tesseract::Tesseract::new(None, Some("eng")).map_err(|e| {
            log::error!("Failed to initialize Tesseract: {}", e.to_string());
            format!("Tesseract init failed: {}. Ensure Tesseract OCR is installed and in PATH.", e.to_string())
        })?
        .set_image(&temp_file_path.to_string_lossy()).map_err(|e| {
            log::error!("Tesseract: Failed to set image: {}", e.to_string());
            format!("Tesseract failed to set image: {}", e.to_string())
        })?
        .set_variable("tessedit_char_whitelist", "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[]^_`{|}~ ").map_err(|e| {
            log::error!("Tesseract: Failed critically setting char whitelist: {}", e.to_string());
            format!("Tesseract failed to set variable: {}\nEnsure whitelist characters are valid.", e.to_string())
        })?
        .get_text().map_err(|e| {
            log::error!("Tesseract: Failed to get text: {}", e.to_string());
            format!("Tesseract failed to get text: {}", e.to_string())
        })
    }();

    // Cleanup of the temporary file saved by ocr_image_buffer
    if let Err(e) = fs::remove_file(&temp_file_path) {
        log::warn!(
            "Failed to remove temporary OCR file {:?}: {}",
            temp_file_path,
            e
        );
    } else {
        log::info!("Temporary OCR file removed: {:?}", temp_file_path);
    }

    match ocr_text_result {
        Ok(text) => {
            log::info!(
                "OCR successful. Text found (first 150 chars): {:.150}",
                text.replace("\n", " ")
            );
            Ok(text)
        }
        Err(e) => Err(e),
    }
}

// --- Tauri Commands ---

#[tauri::command]
async fn capture_interactive_and_ocr(app_handle: AppHandle) -> Result<CaptureResult, String> {
    log::info!("'capture_interactive_and_ocr' command invoked.");

    let temp_image_path: PathBuf;
    let successful_capture: bool; // Track if capture itself succeeded

    #[cfg(target_os = "macos")]
    {
        log::info!("Using 'screencapture -i' on macOS.");
        let temp_dir = env::temp_dir();
        temp_image_path = temp_dir.join(format!("{}.png", Uuid::new_v4().to_string()));
        let capture_status = Command::new("screencapture")
            .arg("-i") // Interactive mode
            .arg(&temp_image_path)
            .status()
            .map_err(|e| format!("Failed to execute screencapture: {}", e))?;

        if !capture_status.success() {
            let err_msg = "screencapture command failed or was cancelled.".to_string();
            log::error!("{}", err_msg);
            return Err(err_msg);
        }
        if !temp_image_path.exists() {
            // This can happen if the user cancels the selection (e.g., presses Esc)
            let err_msg =
                "Interactive screenshot cancelled by user (no image file created).".to_string();
            log::info!("{}", err_msg);
            return Err(err_msg);
        }
        log::info!(
            "Screenshot saved via screencapture to: {:?}",
            temp_image_path
        );
        successful_capture = true; // Mark capture as successful
    }

    #[cfg(target_os = "windows")]
    {
        log::info!("Using Snipping Tool on Windows.");
        // Snipping Tool with /clip copies to clipboard. We then save from clipboard.
        // First, clear clipboard to ensure we get the new snip (optional, but safer)
        // if let Ok(mut ctx) = Clipboard::new() {
        //     let _ = ctx.clear();
        // }

        let capture_process = Command::new("snippingtool.exe")
            .arg("/clip") // This mode copies to clipboard and exits
            .spawn(); // Use spawn to not wait for it if it hangs, but we need to wait briefly.

        match capture_process {
            Ok(mut child) => {
                // Wait for a short period for snipping tool to launch and user to snip.
                // This is a bit of a hack. A more robust solution would involve more complex Windows API interaction.
                thread::sleep(Duration::from_millis(500)); // Give it time to start
                match child.try_wait() {
                    Ok(Some(status)) => log::info!("Snipping Tool exited with: {}", status),
                    Ok(None) => {
                        log::info!(
                            "Snipping Tool still running, user is likely selecting. Polling..."
                        );
                        // Poll for a few seconds for the process to exit
                        for _ in 0..20 {
                            // Poll for up to 10 seconds (20 * 500ms)
                            thread::sleep(Duration::from_millis(500));
                            if let Ok(Some(status)) = child.try_wait() {
                                log::info!("Snipping Tool exited with: {}", status);
                                break;
                            }
                        }
                        // If still running, it might be stuck or user is very slow. Kill it.
                        if child.try_wait().map_or(true, |s| s.is_none()) {
                            log::warn!("Snipping tool seems to be taking too long or is stuck. Attempting to kill.");
                            let _ = child.kill();
                        }
                    }
                    Err(e) => log::warn!("Error waiting for snipping tool: {}", e),
                }
            }
            Err(e) => {
                let err_msg = format!(
                    "Failed to start snippingtool.exe: {}. Make sure it is available.",
                    e
                );
                log::error!("{}", err_msg);
                return Err(err_msg);
            }
        }

        // Try to get image from clipboard
        log::info!("Attempting to retrieve image from clipboard...");
        let mut clipboard = Clipboard::new()
            .map_err(|e| format!("Failed to access clipboard: {}", e.to_string()))?;
        match clipboard.get_image() {
            Ok(image_data) => {
                log::info!(
                    "Image retrieved from clipboard. Width: {}, Height: {}",
                    image_data.width,
                    image_data.height
                );
                let temp_dir = env::temp_dir();
                temp_image_path = temp_dir.join(format!("{}.png", Uuid::new_v4().to_string()));

                // Convert arboard::ImageData to image::DynamicImage
                let img = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
                    image_data.width as u32,
                    image_data.height as u32,
                    image_data.bytes.into_owned(),
                )
                .ok_or_else(|| "Failed to create image buffer from clipboard data".to_string())?;
                let dynamic_img = DynamicImage::ImageRgba8(img);

                dynamic_img
                    .save_with_format(&temp_image_path, ImageFormat::Png)
                    .map_err(|e| format!("Failed to save clipboard image to temp file: {}", e))?;
                log::info!("Clipboard image saved to: {:?}", temp_image_path);
                successful_capture = true; // Mark capture as successful
            }
            Err(e) => {
                let err_msg = format!("Failed to get image from clipboard (Snipping Tool might have been cancelled or no image was copied): {}", e.to_string());
                log::error!("{}", err_msg);
                // Check if snipping tool has a different path for /rect on newer windows versions, this is a common fallback
                // If the error suggests 'NoImage' or similar, it's likely cancellation.
                if e.to_string().contains("No image available") {
                    // Specific check for arboard error
                    let err_msg =
                        "Snipping cancelled or no image data found on clipboard.".to_string();
                    log::info!("{}", err_msg);
                    return Err(err_msg);
                }
                return Err(err_msg);
            }
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let err_msg = "Interactive screenshot is not supported on this OS.".to_string();
        log::error!("{}", err_msg);
        return Err(err_msg);
    }

    // --- Image Loading, OCR, and Base64 Encoding ---
    let ocr_text: String;
    let mut image_base64: Option<String> = None;
    let temp_path_string = temp_image_path.to_string_lossy().to_string(); // Store path for return

    if successful_capture {
        log::info!("Loading image for OCR from: {:?}", temp_image_path);
        match image::open(&temp_image_path) {
            Ok(image_data) => {
                // Perform OCR
                match ocr_image_buffer(&app_handle, &image_data) {
                    Ok(text) => ocr_text = text,
                    Err(e) => {
                        log::warn!("OCR failed after successful capture: {}", e);
                        // Proceed without OCR text, but keep the image
                        ocr_text = "".to_string(); // Ensure it's an empty string not an error propagation
                    }
                }

                // Encode image to Base64 PNG
                log::info!("Encoding image to base64...");
                let mut image_bytes: Vec<u8> = Vec::new();
                match image_data.write_to(&mut Cursor::new(&mut image_bytes), ImageFormat::Png) {
                    Ok(_) => {
                        image_base64 = Some(general_purpose::STANDARD.encode(&image_bytes));
                        log::info!("Image successfully encoded to base64.");
                    }
                    Err(e) => {
                        log::error!("Failed to encode image to PNG bytes for base64: {}", e);
                        // Keep ocr_text if available, but base64 will be None
                    }
                }
            }
            Err(e) => {
                let err_msg = format!(
                    "Failed to load screenshot image from path {:?}: {}",
                    temp_image_path, e
                );
                log::error!("{}", err_msg);
                // Don't return Err here, allow returning partial result if OCR somehow succeeded before (unlikely)
                // or just return empty result. Let's return an empty result for consistency.
                ocr_text = "".to_string();
                image_base64 = None;
            }
        }
    } else {
        // This case should ideally be caught by earlier returns, but as a safeguard:
        log::warn!("Reached post-capture processing without a successful capture flag.");
        ocr_text = "".to_string();
        image_base64 = None;
    }

    // --- IMPORTANT: DO NOT DELETE temp_image_path here ---
    // The frontend will call cleanup_temp_screenshot later when needed.

    // The temp file created *inside* ocr_image_buffer is still cleaned up within that function.

    Ok(CaptureResult {
        ocr_text,
        image_base64,
        temp_path: if successful_capture {
            Some(temp_path_string)
        } else {
            None
        },
    })
}

#[tauri::command]
fn cleanup_temp_screenshot(path: String) -> Result<(), String> {
    log::info!(
        "'cleanup_temp_screenshot' command invoked for path: {}",
        path
    );
    let temp_path = PathBuf::from(path);
    if temp_path.exists() {
        match fs::remove_file(&temp_path) {
            Ok(_) => {
                log::info!(
                    "Successfully removed temporary screenshot file: {:?}",
                    temp_path
                );
                Ok(())
            }
            Err(e) => {
                let err_msg = format!(
                    "Failed to remove temporary screenshot file {:?}: {}",
                    temp_path, e
                );
                log::error!("{}", err_msg);
                Err(err_msg)
            }
        }
    } else {
        log::warn!(
            "Temporary screenshot file not found for cleanup (already deleted?): {:?}",
            temp_path
        );
        Ok(()) // Not an error if it's already gone
    }
}

// --- Other Tauri Commands (send_text_to_model, get_api_key, etc.) should remain the same ---
#[tauri::command]
async fn send_text_to_model(
    messages: Vec<ChatMessage>,
    app_handle: AppHandle,
    window: Window,
) -> Result<(), String> {
    let config = load_config(&app_handle)?;

    let model_name = config.selected_model.clone().unwrap_or_else(|| {
        log::warn!(
            "No model selected in config, using default: {}",
            DEFAULT_MODEL
        );
        DEFAULT_MODEL.to_string()
    });

    log::info!("Processing request for model: {}", model_name);

    // Check if the model is a Gemini model
    if model_name.starts_with("gemini-") || model_name.starts_with("google/") { // Crude check, refine as needed
        let gemini_api_key = match config.gemini_api_key {
            Some(key) if !key.is_empty() => key,
            _ => {
                log::error!("Gemini API key is not set in config for model: {}", model_name);
                return Err(
                    "Gemini API key is not configured. Please set it in settings.".to_string(),
                );
            }
        };
        log::info!("Using Gemini API for model: {}", model_name);

        // TODO: Implement streaming for Gemini API. For now, it will error or not stream.
        // For simplicity in this step, we'll let Gemini calls potentially fail or return non-streamed if they don't support it yet.
        // A proper implementation would require call_gemini_api to also accept window and stream.
        match call_gemini_api(messages, gemini_api_key, model_name.replace("google/", ""), window.clone()).await {
            Ok(_) => Ok(()),
            Err(e) => {
                let _ = window.emit("STREAM_ERROR", StreamErrorPayload { error: e.clone() });
                Err(e)
            }
        }
    } else {
        // Fallback to OpenRouter for other models
        let api_key = match config.api_key {
            Some(key) if !key.is_empty() => key,
            _ => {
                log::error!("OpenRouter API key is not set in config for model: {}", model_name);
                return Err(
                    "OpenRouter API key is not configured. Please set it in settings.".to_string(),
                );
            }
        };
        log::info!(
            "Using OpenRouter API for model: {}. Default model was: {}",
            model_name,
            DEFAULT_MODEL
        );
        match call_openrouter_api(messages, api_key, model_name, window.clone()).await {
            Ok(_) => Ok(()),
            Err(e) => {
                let _ = window.emit("STREAM_ERROR", StreamErrorPayload { error: e.clone() });
                Err(e)
            }
        }
    }
}

#[tauri::command]
async fn get_api_key(app_handle: AppHandle) -> Result<String, String> {
    load_config(&app_handle).map(|config| config.api_key.unwrap_or_default())
}

#[tauri::command]
async fn set_api_key(key: String, app_handle: AppHandle) -> Result<(), String> {
    let mut config = load_config(&app_handle).unwrap_or_else(|e| {
        log::warn!(
            "Failed to load config when setting API key: {}. Using default.",
            e
        );
        AppConfig::default()
    });
    config.api_key = Some(key);
    save_config(&app_handle, &config)
}

#[tauri::command]
async fn get_selected_model(app_handle: AppHandle) -> Result<String, String> {
    load_config(&app_handle).map(|config| {
        config
            .selected_model
            .unwrap_or_else(|| DEFAULT_MODEL.to_string())
    })
}

#[tauri::command]
async fn set_selected_model(model_name: String, app_handle: AppHandle) -> Result<(), String> {
    let allowed_models = vec![
        "deepseek/deepseek-chat-v3-0324:free",
        "deepseek/deepseek-r1:free",
        "gemini-2.0-flash",
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-flash-preview-04-17#thinking-enabled"
    ];
    // Updated check to be more specific
    if !allowed_models.contains(&model_name.as_str()) {
        log::error!("Attempted to set invalid model: {}", model_name);
        return Err(format!("Invalid model selection: {}. Allowed models are: {:?}", model_name, allowed_models));
    }
    let mut config = load_config(&app_handle).unwrap_or_else(|e| {
        log::warn!(
            "Failed to load config when setting model: {}. Using default.",
            e
        );
        AppConfig::default()
    });
    log::info!("Setting selected model to: {}", model_name);
    config.selected_model = Some(model_name);
    save_config(&app_handle, &config)
}

// --- Commands for Gemini API Key ---
#[tauri::command]
async fn get_gemini_api_key(app_handle: AppHandle) -> Result<String, String> {
    load_config(&app_handle).map(|config| config.gemini_api_key.unwrap_or_default())
}

#[tauri::command]
async fn set_gemini_api_key(key: String, app_handle: AppHandle) -> Result<(), String> {
    let mut config = load_config(&app_handle).unwrap_or_else(|e| {
        log::warn!(
            "Failed to load config when setting Gemini API key: {}. Using default.",
            e
        );
        AppConfig::default()
    });
    config.gemini_api_key = Some(key);
    save_config(&app_handle, &config)
}

// --- API Call Logic ---
async fn call_gemini_api(
    messages: Vec<ChatMessage>,
    api_key: String,
    model_identifier_from_config: String, // RENAMED for clarity
    window: Window,
) -> Result<(), String> {
    let client = reqwest::Client::new();

    // MODIFIED: Logic to handle model identifier and generation_config
    let mut actual_model_name_for_api = model_identifier_from_config.clone();
    let mut gen_config: Option<GenerationConfigForGemini> = None;

    if model_identifier_from_config == "gemini-2.5-flash-preview-04-17" {
        // This is the "Gemini 2.5 Flash" (non-thinking explicit budget 0)
        gen_config = Some(GenerationConfigForGemini {
            thinking_config: Some(ThinkingConfig {
                include_thoughts: None, // Let API decide default or if it's implied by budget
                thinking_budget: Some(0),
            }),
            // ..Default::default() // for other potential future fields in GenerationConfigForGemini
        });
        // actual_model_name_for_api is already correct
    } else if model_identifier_from_config == "gemini-2.5-flash-preview-04-17#thinking-enabled" {
        // This is "Gemini 2.5 Flash (Thinking)" (default thinking, no specific budget)
        actual_model_name_for_api = "gemini-2.5-flash-preview-04-17".to_string(); // Use base model name for API
        gen_config = Some(GenerationConfigForGemini {
            thinking_config: None, // No specific thinking_config, so model uses its defaults.
            // This means neither include_thoughts nor thinking_budget will be sent.
            // ..Default::default()
        });
    }
    // For other gemini models, gen_config remains None, and no specific generation_config will be sent.

    let api_url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?key={}&alt=sse",
        actual_model_name_for_api, // Use the potentially modified model name
        api_key
    );

    let request_payload = GeminiChatCompletionRequest {
        contents: messages.into_iter().map(|msg| {
            let role_for_gemini = if msg.role == "assistant" {
                "model".to_string()
            } else {
                msg.role // Assuming it's "user"
            };
            GeminiContent {
                parts: vec![GeminiPart { text: msg.content }],
                role: Some(role_for_gemini),
            }
        }).collect(),
        generation_config: gen_config, // Set the generation_config
    };

    log::info!(
        "Sending STREAMING request to Gemini API for model: {} (API model: {}). Payload: {:?}",
        model_identifier_from_config, actual_model_name_for_api, request_payload
    );

    let response_result = client.post(&api_url)
        .header("Content-Type", "application/json")
        .json(&request_payload)
        .send()
        .await;

    match response_result {
        Ok(response) => {
            if response.status().is_success() {
                use futures_util::StreamExt;
                let mut stream = response.bytes_stream();
                let mut accumulated_content = String::new();
                let mut line_buffer = String::new(); // To handle multi-byte UTF-8 chars split across chunks

                while let Some(item) = stream.next().await {
                    match item {
                        Ok(chunk_bytes) => {
                            match std::str::from_utf8(&chunk_bytes) {
                                Ok(chunk_str) => {
                                    line_buffer.push_str(chunk_str);

                                    // Process complete lines from the buffer
                                    while let Some(newline_pos) = line_buffer.find("\n") {
                                        let line = line_buffer.drain(..newline_pos + 1).collect::<String>();
                                        let trimmed_line = line.trim();

                                        if trimmed_line.starts_with("data: ") {
                                            let data_json_str = &trimmed_line[6..]; // Skip "data: "
                                            // Gemini stream might send an array of responses, often with one element.
                                            // And sometimes it sends a single JSON object directly.
                                            // We need to handle both cases.
                                            // The API doc (and community post) suggests each SSE event is one JSON object representing a GeminiChatCompletionResponse.

                                            // Attempt to parse as a single GeminiChatCompletionResponse
                                            match serde_json::from_str::<GeminiChatCompletionResponse>(data_json_str) {
                                                Ok(gemini_response_chunk) => {
                                                    if let Some(candidate) = gemini_response_chunk.candidates.get(0) {
                                                        if let Some(part) = candidate.content.parts.get(0) {
                                                            let delta = &part.text;
                                                            accumulated_content.push_str(delta);
                                                            if let Err(e) = window.emit("STREAM_CHUNK", StreamChunkPayload { delta: Some(delta.clone()) }) {
                                                                log::error!("Failed to emit STREAM_CHUNK for Gemini: {}", e);
                                                            }
                                                        }
                                                    }
                                                }
                                                Err(e) => {
                                                    // It might be an array of these objects, though less common for pure SSE streams.
                                                    // The official docs for streamGenerateContent show each event as *one* GenerateContentResponse.
                                                    // So, if direct parsing fails, it's likely an error or an unexpected format.
                                                    if !data_json_str.is_empty() && data_json_str != "[" && data_json_str != "]" { // Avoid logging for simple array brackets if they appear alone.
                                                        log::warn!(
                                                            "Failed to parse Gemini stream data JSON as single object: {}. Raw: '{}'",
                                                            e,
                                                            data_json_str
                                                        );
                                                    }
                                                }
                                            }
                                        } else if !trimmed_line.is_empty() {
                                            // Log unexpected non-empty lines that don't start with "data: "
                                            log::warn!("Unexpected line in Gemini stream: {}", trimmed_line);
                                        }
                                    }
                                }
                                Err(e) => {
                                    log::error!("Gemini stream chunk not valid UTF-8: {}", e);
                                    let _ = window.emit("STREAM_ERROR", StreamErrorPayload { error: format!("Gemini stream chunk not valid UTF-8: {}", e) });
                                    return Err(format!("Gemini stream chunk not valid UTF-8: {}", e));
                                }
                            }
                        }
                        Err(e) => {
                            log::error!("Error receiving stream chunk from Gemini: {}", e);
                            let _ = window.emit("STREAM_ERROR", StreamErrorPayload { error: format!("Error in Gemini stream: {}", e) });
                            return Err(format!("Error receiving Gemini stream chunk: {}", e));
                        }
                    }
                }
                // Stream ended
                log::info!("Gemini stream finished. Accumulated content: {}", accumulated_content);
                let _ = window.emit("STREAM_END", StreamEndPayload {
                    full_content: accumulated_content.clone(),
                    reasoning: None, // Gemini API doesn't typically provide separate reasoning field in this way
                });
                Ok(())
            } else {
                let status = response.status();
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Could not read error body from Gemini".to_string());
                log::error!(
                    "Gemini API (streaming) request failed with status {}: {}",
                    status,
                    error_text
                );
                let err_msg = format!("Gemini API (streaming) request failed: {} - {}", status, error_text);
                let _ = window.emit("STREAM_ERROR", StreamErrorPayload { error: err_msg.clone() });
                Err(err_msg)
            }
        }
        Err(e) => {
            log::error!("Network request to Gemini API (streaming) failed: {}", e);
            let err_msg = format!("Gemini API (streaming) network request failed: {}", e);
            let _ = window.emit("STREAM_ERROR", StreamErrorPayload { error: err_msg.clone() });
            Err(err_msg)
        }
    }
}

async fn call_openrouter_api(
    messages: Vec<ChatMessage>,
    api_key: String,
    model_name: String,
    window: Window,
) -> Result<(), String> {
    let client = reqwest::Client::new();
    let api_url = "https://openrouter.ai/api/v1/chat/completions";
    let request_payload = ChatCompletionRequest {
        model: model_name.clone(),
        messages: messages.clone(),
        stream: Some(true),
    };
    log::info!("Sending streaming request to OpenRouter for model: {}", model_name);

    let response_result = client
        .post(api_url)
        .bearer_auth(api_key)
        .header("HTTP-Referer", "http://localhost")
        .header("X-Title", "Shard")
        .json(&request_payload)
        .send()
        .await;

    match response_result {
        Ok(response) => {
            if response.status().is_success() {
                use futures_util::StreamExt; // Import for .next()
                let mut stream = response.bytes_stream();
                let mut accumulated_content = String::new();
                let mut line_buffer = String::new();

                while let Some(item) = stream.next().await {
                    match item {
                        Ok(chunk_bytes) => {
                            match std::str::from_utf8(&chunk_bytes) {
                                Ok(chunk_str) => {
                                    line_buffer.push_str(chunk_str);

                                    // Process complete lines from the buffer
                                    while let Some(newline_pos) = line_buffer.find("\n") {
                                        let line = line_buffer.drain(..newline_pos + 1).collect::<String>();
                                        let trimmed_line = line.trim();

                                        if trimmed_line.starts_with("data: ") {
                                            let data_json_str = &trimmed_line[6..];
                                            if data_json_str == "[DONE]" {
                                                log::info!("OpenRouter stream [DONE] received.");
                                                let _ = window.emit("STREAM_END", StreamEndPayload {
                                                    full_content: accumulated_content.clone(),
                                                    reasoning: None, // TODO: Capture reasoning if available post-stream or via other means
                                                });
                                                return Ok(()); // Successfully finished streaming
                                            }
                                            match serde_json::from_str::<StreamingChatCompletionResponse>(data_json_str) {
                                                Ok(parsed_chunk) => {
                                                    if let Some(choice) = parsed_chunk.choices.get(0) {
                                                        if let Some(content_delta) = &choice.delta.content {
                                                            accumulated_content.push_str(content_delta);
                                                            if let Err(e) = window.emit("STREAM_CHUNK", StreamChunkPayload { delta: Some(content_delta.clone()) }) {
                                                                log::error!("Failed to emit STREAM_CHUNK: {}", e);
                                                            }
                                                        }
                                                    }
                                                }
                                                Err(e) => {
                                                    // Ignore lines that are not valid JSON data chunks, could be comments or empty lines
                                                    if !data_json_str.is_empty() && !data_json_str.starts_with(":") {
                                                        log::warn!("Failed to parse stream data JSON from OpenRouter: '{}'. Raw: '{}'", e, data_json_str);
                                                    }
                                                }
                                            }
                                        } else if !trimmed_line.is_empty() && !trimmed_line.starts_with(":") {
                                            // Log unexpected non-empty, non-comment lines
                                            log::warn!("Unexpected line in OpenRouter stream: {}", trimmed_line);
                                        }
                                    }
                                }
                                Err(e) => {
                                    log::error!("Stream chunk not valid UTF-8: {}", e);
                                    let _ = window.emit("STREAM_ERROR", StreamErrorPayload { error: format!("Stream chunk not valid UTF-8: {}", e) });
                                    return Err(format!("Stream chunk not valid UTF-8: {}", e));
                                }
                            }
                        }
                        Err(e) => {
                            log::error!("Error receiving stream chunk from OpenRouter: {}", e);
                            let _ = window.emit("STREAM_ERROR", StreamErrorPayload { error: format!("Error in stream: {}", e) });
                            return Err(format!("Error receiving stream chunk: {}", e));
                        }
                    }
                }
                // If loop finishes without [DONE], it might be an incomplete stream or an issue.
                // Emit an error or handle as appropriate. For now, assume [DONE] is the primary exit.
                log::warn!("OpenRouter stream ended without [DONE] marker.");
                let _ = window.emit("STREAM_ERROR", StreamErrorPayload { error: "Stream ended without [DONE] marker".to_string() });
                Err("Stream ended without [DONE] marker".to_string())
            } else {
                let status = response.status();
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Could not read error body".to_string());
                log::error!(
                    "OpenRouter API request failed with status {}: {}",
                    status,
                    error_text
                );
                let err_msg = format!("API request failed: {} - {}", status, error_text);
                let _ = window.emit("STREAM_ERROR", StreamErrorPayload { error: err_msg.clone() });
                Err(err_msg)
            }
        }
        Err(e) => {
            log::error!("Network request to OpenRouter failed: {}", e);
            let err_msg = format!("Network request failed: {}", e);
            let _ = window.emit("STREAM_ERROR", StreamErrorPayload { error: err_msg.clone() });
            Err(err_msg)
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let show_hide_modifiers = if cfg!(target_os = "macos") {
        tauri_gs::Modifiers::SUPER | tauri_gs::Modifiers::SHIFT
    } else {
        tauri_gs::Modifiers::CONTROL | tauri_gs::Modifiers::SHIFT
    };
    let show_hide_shortcut_definition = tauri_gs::Shortcut::new(Some(show_hide_modifiers), tauri_gs::Code::KeyZ);

    tauri::Builder::default()
        .plugin(
            tauri_gs::Builder::new()
                .with_handler(move |app_handle: &AppHandle, shortcut_fired: &Shortcut, event: ShortcutEvent| {
                    if shortcut_fired == &show_hide_shortcut_definition {
                        if event.state() == ShortcutState::Pressed {
                            log::info!("[Plugin Shortcut] CmdOrCtrl+Shift+Z pressed. Emitting event to frontend.");
                            app_handle.emit("toggle-main-window", ()).unwrap_or_else(|e| {
                                eprintln!("[Plugin Shortcut] Failed to emit toggle-main-window event: {}", e);
                            });
                        }
                    }
                })
                .build()
        )
        .setup(move |app| {
            #[cfg(desktop)]
            {
                if let Err(e) = app.global_shortcut().register(show_hide_shortcut_definition.clone()) {
                    eprintln!("Failed to register global shortcut via plugin in setup: {}", e);
                } else {
                    log::info!("Successfully registered global shortcut via plugin in setup: CmdOrCtrl+Shift+Z");
                }
            }

            if cfg!(debug_assertions) {
                match app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                    .level(log::LevelFilter::Info)
                    .build(),
                ) {
                    Ok(_) => log::info!("Logger plugin initialized."),
                    Err(e) => eprintln!("Failed to initialize logger plugin: {}", e),
                }
            }
            let main_window = app.get_webview_window("main");
            if let Some(window) = main_window {
                match window.current_monitor() {
                    Ok(Some(monitor)) => {
                        let screen_size = monitor.size();
                        let window_size = window.outer_size().unwrap_or_else(|_| window.inner_size().expect("Failed to get window size"));
                        let new_y = screen_size.height.saturating_sub(window_size.height);
                        match window.set_position(PhysicalPosition::new(0.0, new_y as f64)) {
                            Ok(_) => log::info!("Window positioned to bottom-left (0, {})", new_y),
                            Err(e) => log::error!("Failed to set window position: {}", e),
                        }
                    }
                    Ok(None) => log::error!("Could not get current monitor info."),
                    Err(e) => log::error!("Error getting monitor info: {}", e),
                }
            } else {
                log::error!("Could not get main window to set position.");
            }
            let config_handle = app.handle().clone();
            match load_config(&config_handle) {
                Ok(config) => {
                    log::info!(
                        "Loaded config during setup. API key is {}. Selected model: {:?}. Gemini API key is {}.",
                        if config.api_key.is_some() { "set" } else { "not set" },
                        config.selected_model.as_deref().unwrap_or("None (will use default)"),
                        if config.gemini_api_key.is_some() { "set" } else { "not set" }
                    );
                    let config_path = get_config_path(&config_handle).expect("Failed to get config path in setup");
                    if config_path.exists() && config.selected_model.is_none() {
                        log::info!("Existing config file found without a selected model. Saving default model selection.");
                        let mut updated_config = config.clone();
                        updated_config.selected_model = Some(DEFAULT_MODEL.to_string());
                        if let Err(e) = save_config(&config_handle, &updated_config) {
                            log::error!("Failed to save default model to existing config: {}", e);
                        } else {
                            log::info!("Saved default model selection to existing config file.");
                        }
                    } else if !config_path.exists() {
                        log::info!("No config file found. Saving initial default config.");
                        let mut default_config = AppConfig::default();
                        default_config.selected_model = Some(DEFAULT_MODEL.to_string());
                        if let Err(e) = save_config(&config_handle, &default_config) {
                            log::error!("Failed to save initial default config: {}", e);
                        } else {
                            log::info!("Saved initial default config file.");
                        }
                    }
                }
                Err(e) => {
                    log::error!("Failed to load config during setup: {}. Creating default.", e);
                    let mut default_config = AppConfig::default();
                    default_config.selected_model = Some(DEFAULT_MODEL.to_string());
                    if let Err(save_err) = save_config(&config_handle, &default_config) {
                        log::error!("Failed to save default config after load error: {}", save_err);
                    } else {
                        log::info!("Saved default config file because initial load failed.");
                    }
                }
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            match event {
                WindowEvent::CloseRequested { api, .. } => {
                    if let Err(e) = window.hide() {
                        eprintln!("Failed to hide window on close request: {}", e);
                    }
                    api.prevent_close();
                }
                _ => {}
            }
        })
        .invoke_handler(tauri::generate_handler![
            send_text_to_model,
            get_api_key,
            set_api_key,
            get_selected_model,
            set_selected_model,
            capture_interactive_and_ocr,
            cleanup_temp_screenshot,
            get_gemini_api_key,
            set_gemini_api_key
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
