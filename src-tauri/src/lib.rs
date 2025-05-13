use serde::{Deserialize, Serialize};
use reqwest;
use tauri::{AppHandle, Manager};
use std::fs;
use std::path::PathBuf;
use serde_json;
use tauri::PhysicalPosition;
use image::{DynamicImage, ImageFormat};
use std::env; // For temp_dir
use uuid::Uuid; // For unique filenames
use tesseract;   // Uncommented
use std::process::Command;
use base64::{engine::general_purpose, Engine as _}; // Added base64 import
use std::io::Cursor; // Added for base64 encoding

#[cfg(target_os = "windows")]
use arboard::Clipboard;
#[cfg(target_os = "windows")]
use std::time::Duration;
#[cfg(target_os = "windows")]
use std::thread;

// Default model if none is selected
const DEFAULT_MODEL: &str = "deepseek/deepseek-chat-v3-0324:free";

// --- Config Structures ---
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
struct AppConfig {
    api_key: Option<String>,
    selected_model: Option<String>,
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
        log::info!("Config file not found at {:?}, returning default.", config_path);
        return Ok(AppConfig::default());
    }
    log::info!("Loading config from {:?}", config_path);
    let content = fs::read_to_string(&config_path)
        .map_err(|e| format!("Failed to read config file: {}", e))?;
    toml::from_str(&content)
        .map_err(|e| format!("Failed to parse config file: {}", e))
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
    let toml_string = toml::to_string_pretty(config)
        .map_err(|e| format!("Failed to serialize config: {}", e))?;
    fs::write(&config_path, toml_string)
        .map_err(|e| format!("Failed to write config file: {}", e))
}

// Request Structures
#[derive(Serialize, Deserialize, Debug)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
}

// Response Structures
#[derive(Deserialize, Debug)]
struct ChoiceMessage {
    content: String,
    reasoning: Option<String>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    message: ChoiceMessage,
}

// Define a structure to return both content and reasoning
#[derive(Serialize, Deserialize, Debug)]
struct ModelResponse {
    content: String,
    reasoning: Option<String>,
}

#[derive(Deserialize, Debug)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
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
        fs::create_dir_all(&temp_dir).map_err(|e| format!("Failed to create cache directory: {}", e))?;
    }
    let temp_filename = format!("{}.png", Uuid::new_v4().to_string());
    let temp_file_path = temp_dir.join(&temp_filename);

    log::info!("Saving image to temporary file for OCR: {:?}", temp_file_path);
    img_buffer.save_with_format(&temp_file_path, ImageFormat::Png).map_err(|e| {
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
        log::warn!("Failed to remove temporary OCR file {:?}: {}", temp_file_path, e);
    } else {
        log::info!("Temporary OCR file removed: {:?}", temp_file_path);
    }

    match ocr_text_result {
        Ok(text) => {
            log::info!("OCR successful. Text found (first 150 chars): {:.150}", text.replace("\n", " "));
            Ok(text)
        }
        Err(e) => Err(e)
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
            let err_msg = "Interactive screenshot cancelled by user (no image file created).".to_string();
            log::info!("{}", err_msg);
            return Err(err_msg);
        }
        log::info!("Screenshot saved via screencapture to: {:?}", temp_image_path);
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
                        log::info!("Snipping Tool still running, user is likely selecting. Polling...");
                        // Poll for a few seconds for the process to exit
                        for _ in 0..20 { // Poll for up to 10 seconds (20 * 500ms)
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
                let err_msg = format!("Failed to start snippingtool.exe: {}. Make sure it is available.", e);
                log::error!("{}", err_msg);
                return Err(err_msg);
            }
        }

        // Try to get image from clipboard
        log::info!("Attempting to retrieve image from clipboard...");
        let mut clipboard = Clipboard::new().map_err(|e| format!("Failed to access clipboard: {}", e.to_string()))?;
        match clipboard.get_image() {
            Ok(image_data) => {
                log::info!("Image retrieved from clipboard. Width: {}, Height: {}", image_data.width, image_data.height);
                let temp_dir = env::temp_dir();
                temp_image_path = temp_dir.join(format!("{}.png", Uuid::new_v4().to_string()));

                // Convert arboard::ImageData to image::DynamicImage
                let img = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(image_data.width as u32, image_data.height as u32, image_data.bytes.into_owned())
                    .ok_or_else(|| "Failed to create image buffer from clipboard data".to_string())?;
                let dynamic_img = DynamicImage::ImageRgba8(img);

                dynamic_img.save_with_format(&temp_image_path, ImageFormat::Png).map_err(|e| {
                    format!("Failed to save clipboard image to temp file: {}", e)
                })?;
                log::info!("Clipboard image saved to: {:?}", temp_image_path);
                successful_capture = true; // Mark capture as successful
            }
            Err(e) => {
                let err_msg = format!("Failed to get image from clipboard (Snipping Tool might have been cancelled or no image was copied): {}", e.to_string());
                log::error!("{}", err_msg);
                // Check if snipping tool has a different path for /rect on newer windows versions, this is a common fallback
                // If the error suggests 'NoImage' or similar, it's likely cancellation.
                 if e.to_string().contains("No image available") { // Specific check for arboard error
                    let err_msg = "Snipping cancelled or no image data found on clipboard.".to_string();
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
                let err_msg = format!("Failed to load screenshot image from path {:?}: {}", temp_image_path, e);
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
        temp_path: if successful_capture { Some(temp_path_string) } else { None }
    })
}

#[tauri::command]
fn cleanup_temp_screenshot(path: String) -> Result<(), String> {
    log::info!("'cleanup_temp_screenshot' command invoked for path: {}", path);
    let temp_path = PathBuf::from(path);
    if temp_path.exists() {
        match fs::remove_file(&temp_path) {
            Ok(_) => {
                log::info!("Successfully removed temporary screenshot file: {:?}", temp_path);
                Ok(())
            }
            Err(e) => {
                let err_msg = format!("Failed to remove temporary screenshot file {:?}: {}", temp_path, e);
                log::error!("{}", err_msg);
                Err(err_msg)
            }
        }
    } else {
        log::warn!("Temporary screenshot file not found for cleanup (already deleted?): {:?}", temp_path);
        Ok(()) // Not an error if it's already gone
    }
}

// --- Other Tauri Commands (send_text_to_model, get_api_key, etc.) should remain the same ---
#[tauri::command]
async fn send_text_to_model(text: String, app_handle: AppHandle) -> Result<ModelResponse, String> {
    let config = load_config(&app_handle)?;
    let api_key = match config.api_key {
        Some(key) if !key.is_empty() => key,
        _ => {
            log::error!("API key is not set in config.");
            return Err("API key is not configured. Please set it in the application settings.".to_string());
        }
    };
    let model_name = config.selected_model.unwrap_or_else(|| {
        log::warn!("No model selected in config, using default: {}", DEFAULT_MODEL);
        DEFAULT_MODEL.to_string()
    });
    log::info!("Using model: {}", model_name);
    call_openrouter_api(text, api_key, model_name).await
}

#[tauri::command]
async fn get_api_key(app_handle: AppHandle) -> Result<String, String> {
    load_config(&app_handle)
        .map(|config| config.api_key.unwrap_or_default())
}

#[tauri::command]
async fn set_api_key(key: String, app_handle: AppHandle) -> Result<(), String> {
    let mut config = load_config(&app_handle).unwrap_or_else(|e| {
        log::warn!("Failed to load config when setting API key: {}. Using default.", e);
        AppConfig::default()
    });
    config.api_key = Some(key);
    save_config(&app_handle, &config)
}

#[tauri::command]
async fn get_selected_model(app_handle: AppHandle) -> Result<String, String> {
    load_config(&app_handle)
        .map(|config| config.selected_model.unwrap_or_else(|| DEFAULT_MODEL.to_string()))
}

#[tauri::command]
async fn set_selected_model(model_name: String, app_handle: AppHandle) -> Result<(), String> {
    let allowed_models = vec![
        "google/gemini-2.0-flash-exp:free",
        "deepseek/deepseek-chat-v3-0324:free",
        "deepseek/deepseek-r1:free",
    ];
    if !allowed_models.contains(&model_name.as_str()) {
        log::error!("Attempted to set invalid model: {}", model_name);
        return Err(format!("Invalid model selection: {}", model_name));
    }
    let mut config = load_config(&app_handle).unwrap_or_else(|e| {
        log::warn!("Failed to load config when setting model: {}. Using default.", e);
        AppConfig::default()
    });
    log::info!("Setting selected model to: {}", model_name);
    config.selected_model = Some(model_name);
    save_config(&app_handle, &config)
}

// --- API Call Logic --- (Should remain the same)
async fn call_openrouter_api(user_text: String, api_key: String, model_name: String) -> Result<ModelResponse, String> {
    let client = reqwest::Client::new();
    let api_url = "https://openrouter.ai/api/v1/chat/completions";
    let request_payload = ChatCompletionRequest {
        model: model_name.clone(),
        messages: vec![
            ChatMessage { role: "system".to_string(), content: "You are a helpful assistant.".to_string() },
            ChatMessage { role: "user".to_string(), content: user_text },
        ],
    };
    log::info!("Sending request to OpenRouter for model: {}", model_name);
    match client.post(api_url)
        .bearer_auth(api_key)
        .header("HTTP-Referer", "http://localhost")
        .header("X-Title", "Shard")
        .json(&request_payload)
        .send()
        .await {
            Ok(response) => {
                if response.status().is_success() {
                    match response.text().await {
                        Ok(raw_json_text) => {
                            log::debug!("Raw JSON response from OpenRouter: {}", raw_json_text);
                            match serde_json::from_str::<ChatCompletionResponse>(&raw_json_text) {
                                Ok(chat_response) => {
                                    if let Some(choice) = chat_response.choices.get(0) {
                                        Ok(ModelResponse {
                                            content: choice.message.content.clone(),
                                            reasoning: choice.message.reasoning.clone(),
                                        })
                                    } else {
                                        log::error!("OpenRouter response contained no choices after parsing.");
                                        Err("No response choices received from model".to_string())
                                    }
                                }
                                Err(e) => {
                                     log::error!("Failed to parse JSON (from_str): {}", e);
                                     log::error!("Raw JSON causing parse error: {}", raw_json_text);
                                     Err(format!("Failed to parse response JSON: {}", e))
                                }
                            }
                        }
                        Err(e) => {
                            log::error!("Failed to read response text: {}", e);
                            Err(format!("Failed to read response text: {}", e))
                        }
                    }
                } else {
                    let status = response.status();
                    let error_text = response.text().await.unwrap_or_else(|_| "Could not read error body".to_string());
                    log::error!("OpenRouter API request failed with status {}: {}", status, error_text);
                    Err(format!("API request failed: {} - {}", status, error_text))
                }
            }
            Err(e) => {
                 log::error!("Network request to OpenRouter failed: {}", e);
                 Err(format!("Network request failed: {}", e))
            }
        }
}


#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        // Removed region selector plugin logic if any was here
        .setup(move |app| {
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
                        "Loaded config during setup. API key is {}. Selected model: {:?}",
                        if config.api_key.is_some() { "set" } else { "not set" },
                        config.selected_model.as_deref().unwrap_or("None (will use default)")
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
        .invoke_handler(tauri::generate_handler![
            send_text_to_model,
            get_api_key,
            set_api_key,
            get_selected_model,
            set_selected_model,
            capture_interactive_and_ocr, // Updated command name
            cleanup_temp_screenshot    // Added new command
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
