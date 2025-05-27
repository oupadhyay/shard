#![allow(unexpected_cfgs)] // Added to suppress unexpected_cfgs warnings from dependencies

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
use tauri::{AppHandle, Emitter, Manager, Window, WindowEvent}; // Added Emitter and Window
use tauri_nspanel::WebviewWindowExt; // CORRECTED IMPORT
use tauri_nspanel::{panel_delegate, Panel}; // Added for panel delegate
use tauri_plugin_global_shortcut::{
    self as tauri_gs, GlobalShortcutExt, Shortcut, ShortcutEvent, ShortcutState,
};
use tesseract; // Uncommented
use time::OffsetDateTime;
use uuid::Uuid; // For unique filenames // Added for base64 encoding // Plugin imports
use yahoo_finance_api as yfa; // Using an alias for brevity // For timestamp conversion

// Default model if none is selected
const DEFAULT_MODEL: &str = "deepseek/deepseek-chat-v3-0324:free";

// --- System Instruction ---
const SYSTEM_INSTRUCTION: &str = "You are a helpful assistant that provides accurate, factual answers. If you do not know the answer, make your best guess. You are business casual in tone and prefer concise responses. Avoid starting responses with \"**\". Prefer bulleted lists when needed but no nested lists/sub-bullets. Use markdown formatting for code blocks and links and $$ for LaTeX equations.";

// --- Config Structures ---
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
struct AppConfig {
    api_key: Option<String>,
    selected_model: Option<String>,
    gemini_api_key: Option<String>,  // Added for Gemini
    enable_web_search: Option<bool>, // ADDED for web search toggle
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
    // log::info!("Loading config from {:?}", config_path);
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
    #[serde(skip_serializing_if = "Option::is_none")]
    include_reasoning: Option<bool>,
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
    role: Option<String>,    // Role might appear in first chunk
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<String>,
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

// --- Web Search Event Payloads ---
#[derive(Serialize, Clone, Debug)]
struct ArticleLookupStartedPayload {
    query: String,
}

#[derive(Serialize, Clone, Debug)]
struct ArticleLookupCompletedPayload {
    query: String,
    success: bool,
    summary: Option<String>,
    source_name: Option<Vec<String>>,
    source_url: Option<Vec<String>>,
    error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IterativeSearchResult {
    pub title: String,
    pub summary: String,
    pub url: String,
    pub path_taken: Vec<String>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "decision_type")]
enum AnalysisLLMDecision {
    #[serde(rename = "FOUND_ANSWER")]
    FoundAnswer { summary: String, title: String },
    #[serde(rename = "NEXT_TERM")]
    NextTerm { term: String, reason: String },
    #[serde(rename = "STOP")]
    Stop { reason: String },
}

// --- ADDED: Weather Lookup Event Payloads ---
#[derive(Serialize, Clone, Debug)]
struct WeatherLookupStartedPayload {
    location: String,
}

#[derive(Serialize, Clone, Debug)]
struct WeatherLookupCompletedPayload {
    location: String,
    success: bool,
    temperature: Option<f32>,
    unit: Option<String>,
    description: Option<String>,
    error: Option<String>,
}

// --- Financial Data Event Payloads ---
#[derive(Serialize, Clone, Debug)]
struct FinancialDataStartedPayload {
    query: String,
    symbol: String,
}

#[derive(Serialize, Clone, Debug)]
struct FinancialDataCompletedPayload {
    query: String,
    symbol: String,
    success: bool,
    data: Option<String>, // Formatted financial data string
    error: Option<String>,
}

// --- ADDED: Wikipedia API Structures ---
#[derive(Serialize, Deserialize, Debug, Clone)]
struct WikipediaQueryPage {
    pageid: Option<i64>,
    title: Option<String>,
    extract: Option<String>,
    missing: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct WikipediaQuery {
    pages: Vec<WikipediaQueryPage>, // Changed from HashMap<String, WikipediaQueryPage>
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct WikipediaResponse {
    batchcomplete: Option<bool>, // Changed from Option<String> to Option<bool>
    query: Option<WikipediaQuery>,
}

// --- ADDED: Open-Meteo Geocoding API Structures ---
#[derive(Serialize, Deserialize, Debug, Clone)]
struct GeocodingResult {
    id: Option<f64>,
    name: Option<String>,
    latitude: Option<f32>,
    longitude: Option<f32>,
    country: Option<String>,
    admin1: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct GeocodingResponse {
    results: Option<Vec<GeocodingResult>>,
    generationtime_ms: Option<f32>,
}

// --- ADDED: Open-Meteo Weather API Structures ---
#[derive(Serialize, Deserialize, Debug, Clone)]
struct WeatherCurrentUnits {
    temperature_2m: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct WeatherCurrentData {
    time: Option<String>,
    interval: Option<i32>,
    temperature_2m: Option<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct WeatherResponse {
    latitude: Option<f32>,
    longitude: Option<f32>,
    generationtime_ms: Option<f32>,
    utc_offset_seconds: Option<i32>,
    timezone: Option<String>,
    timezone_abbreviation: Option<String>,
    elevation: Option<f32>,
    current_units: Option<WeatherCurrentUnits>,
    current: Option<WeatherCurrentData>,
}

fn separate_reasoning_from_content(text: &str) -> (String, String) {
    let mut content_parts = Vec::new();
    let mut reasoning_parts = Vec::new();

    // Split text by reasoning block headers (lines that start and end with **)
    let mut current_section = String::new();
    let mut is_reasoning_section = false;

    for line in text.lines() {
        let trimmed = line.trim();

        // Check if this line is a reasoning block header
        if trimmed.starts_with("**") && trimmed.ends_with("**") && trimmed.len() > 4 {
            // Save the previous section
            if !current_section.trim().is_empty() {
                if is_reasoning_section {
                    reasoning_parts.push(current_section.trim().to_string());
                } else {
                    content_parts.push(current_section.trim().to_string());
                }
            }

            // Start new reasoning section with this header
            current_section = line.to_string() + "\n";
            is_reasoning_section = true;
        } else {
            // Add line to current section
            current_section.push_str(line);
            current_section.push('\n');

            // Check if this looks like the end of reasoning and start of content
            if is_reasoning_section && !trimmed.is_empty() {
                // Simple heuristic: if line doesn't contain reasoning-style language
                // and doesn't start with common reasoning words, it might be content
                let reasoning_indicators = [
                    "I'm",
                    "I've",
                    "My goal",
                    "I will",
                    "I need",
                    "I want",
                    "I think",
                    "I believe",
                    "I should",
                    "I'll",
                ];
                let has_reasoning_language = reasoning_indicators
                    .iter()
                    .any(|&indicator| trimmed.contains(indicator));

                // Also check if this looks like a final answer or regular response
                let looks_like_final_answer = trimmed.len() > 20
                    && !has_reasoning_language
                    && (trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?'));

                if looks_like_final_answer {
                    // This seems to be the start of the actual response
                    // Split the current section
                    let lines_in_section: Vec<&str> = current_section.trim().lines().collect();
                    if lines_in_section.len() > 1 {
                        // Last line is probably content, rest is reasoning
                        let reasoning_part =
                            lines_in_section[..lines_in_section.len() - 1].join("\n");
                        let content_part = lines_in_section.last().unwrap();

                        if !reasoning_part.trim().is_empty() {
                            reasoning_parts.push(reasoning_part.trim().to_string());
                        }

                        current_section = content_part.to_string() + "\n";
                        is_reasoning_section = false;
                    }
                }
            }
        }
    }

    // Add the final section
    if !current_section.trim().is_empty() {
        if is_reasoning_section {
            reasoning_parts.push(current_section.trim().to_string());
        } else {
            content_parts.push(current_section.trim().to_string());
        }
    }

    // Join parts with proper spacing
    let content = content_parts.join("\n\n").trim().to_string();
    let mut reasoning = reasoning_parts.join("\n\n").trim().to_string();

    // Convert **Header** patterns to proper markdown headers with spacing
    // Find any **Header** pattern and replace with proper newlines + ## header
    let re = regex::Regex::new(r"\*\*([^*]+?)\*\*").unwrap();
    reasoning = re.replace_all(&reasoning, "\n\n## $1").to_string();

    // Clean up any multiple newlines and ensure proper start
    reasoning = reasoning.replace("\n\n\n", "\n\n").to_string();

    (content, reasoning)
}

// --- ADDED: Wikipedia Lookup Function ---
async fn perform_wikipedia_lookup(
    client: &reqwest::Client,
    search_term: &str,
) -> Result<Option<(String, String, String)>, String> {
    // (summary, source_name, source_url)
    let base_url = "https://en.wikipedia.org/w/api.php";
    let params = [
        ("action", "query"),
        ("format", "json"),
        ("titles", search_term),
        ("prop", "extracts"),
        ("exintro", "true"),
        ("explaintext", "true"),
        ("redirects", "1"),
        ("formatversion", "2"),
    ];
    let request_url = client
        .get(base_url)
        .query(&params)
        .build()
        .unwrap()
        .url()
        .to_string();
    log::info!("Performing Wikipedia lookup. Request URL: {}", request_url);
    match client.get(base_url).query(&params).send().await {
        Ok(response) => {
            let status = response.status();
            let response_text = response
                .text()
                .await
                .map_err(|e| format!("Wikipedia: Failed to read response text: {}", e))?;
            if status.is_success() {
                match serde_json::from_str::<WikipediaResponse>(&response_text) {
                    Ok(wiki_response) => {
                        log::info!("Wikipedia: Successfully parsed JSON: {:#?}", wiki_response);
                        if let Some(query_data) = wiki_response.query {
                            if let Some(page) = query_data.pages.first() {
                                // Changed from .values().next() to .first()
                                if page.missing.is_some() {
                                    log::info!("Wikipedia: Page '{}' does not exist.", search_term);
                                    return Ok(None);
                                }
                                if let Some(extract) = &page.extract {
                                    if !extract.trim().is_empty() {
                                        let title = page
                                            .title
                                            .clone()
                                            .unwrap_or_else(|| search_term.to_string());
                                        let source_url = format!(
                                            "https://en.wikipedia.org/wiki/{}",
                                            title.replace(" ", "_")
                                        );
                                        log::info!(
                                            "Wikipedia: Found extract for title '{}'",
                                            title
                                        );
                                        return Ok(Some((
                                            title,
                                            extract.trim().to_string(),
                                            source_url,
                                        )));
                                    }
                                }
                            }
                        }
                        log::info!("Wikipedia: No suitable extract for '{}'.", search_term);
                        Ok(None)
                    }
                    Err(e) => {
                        log::error!(
                            "Wikipedia: Failed to parse JSON: {}. Raw: {}",
                            e,
                            response_text
                        );
                        Err(format!(
                            "Wikipedia JSON parse error: {}. Ensure response is valid JSON.",
                            e
                        ))
                    }
                }
            } else {
                log::error!("Wikipedia: API error status {}: {}", status, response_text);
                Err(format!(
                    "Wikipedia API error: {} - {}",
                    status, response_text
                ))
            }
        }
        Err(e) => {
            log::error!("Wikipedia: Network error: {}", e);
            Err(format!("Wikipedia network error: {}", e))
        }
    }
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

// --- ADDED: Helper function to extract stock symbol ---
fn extract_stock_symbol(query: &str) -> Option<String> {
    log::info!(
        "Attempting to extract stock symbol from query: '{}' using ticker-sniffer.",
        query
    );

    // According to ticker-sniffer docs, `is_case_sensitive_doc_parsing = false`
    // might be better for search query inputs, though it can increase false positives
    // between nouns (e.g., "apple") and company names (e.g., "Apple").
    match ticker_sniffer::extract_tickers_from_text(query, true) {
        Ok(ticker_map) => {
            if ticker_map.is_empty() {
                log::warn!("ticker-sniffer found no symbols in query: '{}'", query);
                None
            } else {
                // Find the ticker with the highest frequency.
                // If there are multiple with the same highest frequency, pick the first one alphabetically.
                let mut best_symbol: Option<String> = None;

                // Sort by frequency (desc) then by symbol (asc) for tie-breaking
                let mut sorted_tickers: Vec<(&String, &usize)> = ticker_map.iter().collect();
                sorted_tickers.sort_by(|a, b| {
                    b.1.cmp(a.1) // Sort by frequency descending
                        .then_with(|| a.0.cmp(b.0)) // Then by symbol ascending
                });

                if let Some((symbol, freq)) = sorted_tickers.first() {
                    log::info!(
                        "ticker-sniffer extracted symbol: '{}' with frequency {} from query: '{}' (Full map: {:?})",
                        symbol, freq, query, ticker_map
                    );
                    best_symbol = Some(symbol.to_string());
                }

                best_symbol
            }
        }
        Err(e) => {
            log::error!(
                "ticker-sniffer failed to extract symbols from query '{}': {}",
                query,
                e
            );
            None
        }
    }
}

// --- ADDED: Geocoding Function ---
async fn geocode_location(
    client: &reqwest::Client,
    location_name: &str,
) -> Result<Option<(f32, f32, String)>, String> {
    // (latitude, longitude, resolved_name)
    let base_url = "https://geocoding-api.open-meteo.com/v1/search";
    let params = [
        ("name", location_name),
        ("count", "1"),
        ("language", "en"),
        ("format", "json"),
    ];
    let request_url = client
        .get(base_url)
        .query(&params)
        .build()
        .unwrap()
        .url()
        .to_string();
    log::info!("Geocoding for '{}'. URL: {}", location_name, request_url);
    match client.get(base_url).query(&params).send().await {
        Ok(response) => {
            let status = response.status();
            let response_text = response
                .text()
                .await
                .map_err(|e| format!("Geocoding: Failed to read response text: {}", e))?;
            if status.is_success() {
                match serde_json::from_str::<GeocodingResponse>(&response_text) {
                    Ok(geo_response) => {
                        log::info!("Geocoding: Parsed JSON: {:#?}", geo_response);
                        if let Some(results) = geo_response.results {
                            if let Some(top) = results.first() {
                                if let (Some(lat_val), Some(lon_val), Some(name_val)) =
                                    (top.latitude, top.longitude, &top.name)
                                {
                                    let resolved = format!(
                                        "{}{}{}",
                                        name_val,
                                        top.admin1
                                            .as_ref()
                                            .map_or_else(|| "".to_string(), |a| format!(", {}", a)),
                                        top.country
                                            .as_ref()
                                            .map_or_else(|| "".to_string(), |c| format!(", {}", c))
                                    );
                                    log::info!(
                                        "Geocoding: Found for '{}': ({}, {}). Resolved: {}",
                                        location_name,
                                        lat_val,
                                        lon_val,
                                        resolved
                                    );
                                    return Ok(Some((lat_val, lon_val, resolved)));
                                    // No deref needed for f32
                                }
                            }
                        }
                        log::info!("Geocoding: No coords for '{}'.", location_name);
                        Ok(None)
                    }
                    Err(e) => {
                        log::error!("Geocoding: JSON parse error: {}. Raw: {}", e, response_text);
                        Err(format!(
                            "Geocoding JSON error: {}. Ensure response is valid JSON.",
                            e
                        ))
                    }
                }
            } else {
                log::error!("Geocoding: API error status {}: {}", status, response_text);
                Err(format!(
                    "Geocoding API error: {} - {}",
                    status, response_text
                ))
            }
        }
        Err(e) => {
            log::error!("Geocoding: Network error: {}", e);
            Err(format!("Geocoding network error: {}", e))
        }
    }
}

// --- ADDED: Financial Data Lookup Function ---
async fn perform_financial_data_lookup(
    _client: &reqwest::Client, // Not directly used by yfa, but kept for consistency if other libs need it
    symbol: &str,
) -> Result<String, String> {
    log::info!(
        "Performing financial data lookup for symbol: '{}' using yahoo_finance_api",
        symbol
    );

    let provider = match yfa::YahooConnector::new() {
        Ok(p) => p,
        Err(e) => {
            let err_msg = format!("Failed to create YahooConnector: {}", e.to_string());
            log::error!("{}", err_msg);
            return Err(err_msg);
        }
    };

    match provider.get_latest_quotes(symbol, "1d").await {
        // Get latest daily quote
        Ok(response) => {
            if let Some(quote) = response.last_quote().ok() {
                // last_quote returns Result<Quote, Error>
                // Convert Unix timestamp to readable date
                // The timestamp from yahoo_finance_api::Quote is u64
                let dt = OffsetDateTime::from_unix_timestamp(quote.timestamp as i64)
                    .map_err(|e| format!("Failed to convert timestamp: {}", e))?;

                let date_str = dt
                    .format(
                        &time::format_description::parse("[year]-[month]-[day]")
                            .map_err(|e| format!("Failed to parse date format: {}", e))?,
                    )
                    .map_err(|e| format!("Failed to format date: {}", e))?;

                let formatted_data = format!(
                    "Latest data for {}: Date: {}, Open: {:.2}, High: {:.2}, Low: {:.2}, Close: {:.2}, Volume: {}",
                    symbol.to_uppercase(),
                    date_str,
                    quote.open,
                    quote.high,
                    quote.low,
                    quote.close,
                    quote.volume
                );
                log::info!(
                    "Financial data lookup successful for symbol: '{}'. Data: {}",
                    symbol,
                    formatted_data
                );
                Ok(formatted_data)
            } else {
                let msg = format!("No quote data found for symbol {}.", symbol);
                log::warn!("Financial data lookup for symbol '{}': {}", symbol, msg);
                Err(msg)
            }
        }
        Err(e) => {
            let err_msg = format!(
                "Failed to retrieve financial data for {} from yahoo_finance_api: {}",
                symbol,
                e.to_string()
            );
            log::error!("{}", err_msg);
            Err(err_msg)
        }
    }
}

// --- Tauri Commands ---

#[tauri::command]
fn trigger_backend_window_toggle(app_handle: AppHandle) -> Result<(), String> {
    log::info!("[Backend] trigger_backend_window_toggle called from frontend.");
    app_handle.emit("toggle-main-window", ()).map_err(|e| {
        let err_msg = format!(
            "Failed to emit toggle-main-window event from backend: {}",
            e
        );
        log::error!("{}", err_msg);
        err_msg
    })
}

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
    // Create a new message list, starting with the system instruction.
    let mut final_messages = Vec::new();
    final_messages.push(ChatMessage {
        role: "system".to_string(), // This role will be adapted by call_gemini_api
        content: SYSTEM_INSTRUCTION.to_string(),
    });
    // Original user messages will be added after potential web search or financial data context

    let config = load_config(&app_handle)?;

    let model_name = config.selected_model.clone().unwrap_or_else(|| {
        log::warn!(
            "No model selected in config, using default: {}",
            DEFAULT_MODEL
        );
        DEFAULT_MODEL.to_string()
    });

    log::info!("Processing request for model: {}", model_name);

    // --- Web Search Logic --- (Now Article Lookup)
    let mut article_lookup_performed_successfully = false;
    let mut article_lookup_result_text: Option<String> = None;
    // --- Financial Data Logic ---
    let mut financial_data_fetched_successfully = false;
    let mut financial_data_result_text: Option<String> = None;
    // --- ADDED: Weather Lookup Logic State ---
    let mut weather_lookup_performed_successfully = false;
    let mut weather_lookup_result_text: Option<String> = None;

    // Create reqwest client once
    let client = reqwest::Client::new();

    if config.enable_web_search.unwrap_or(true) {
        if let Some(last_user_message) = messages.last() {
            if last_user_message.role == "user" {
                let user_query = last_user_message.content.trim();
                let query_words: Vec<&str> = user_query.split_whitespace().collect();

                if query_words.len() >= 1 {
                    log::info!(
                        "Considering external data lookup for query: '{}'",
                        user_query
                    );

                    let decider_prompt =
                        "You are an intelligent assistant that categorizes user queries to determine the best data retrieval strategy.\n".to_string() +
                        "Analyze the user\'s query and decide if it primarily requires:\n" +
                        "1. Factual information lookup about a specific topic, person, place, or concept (e.g., definitions, history, general knowledge). If so, respond with only the exact string \"WIKIPEDIA_LOOKUP\".\n" +
                        "   Examples: \"What is photosynthesis?\", \"Tell me about the Eiffel Tower\", \"Who was Marie Curie?\"\n" +
                        "2. Current weather conditions for a specific location. If so, respond with only the exact string \"WEATHER_LOOKUP\".\n" +
                        "   Examples: \"weather in San Francisco\", \"what's the temperature in London?\", \"Is it raining in Tokyo?\"\n" +
                        "3. Specific financial market data for a publicly traded stock or symbol. If so, respond with only the exact string \"FINANCIAL_DATA\".\n" +
                        "   Examples: \"What is the stock price of GOOGL?\", \"AAPL stock quote\"\n" +
                        "4. No external data lookup, meaning the query is conversational, a command, a creative request, or can be answered from general LLM knowledge. If so, respond with only the exact string \"NO_LOOKUP\".\n" +
                        "   Examples: \"Tell me a joke.\", \"Summarize this text for me: ...\", \"What is the capital of Germany?\"\n" +
                        &format!("User query: '{}'\n", user_query) +
                        "Based on this query, respond with one of the exact strings: \"WIKIPEDIA_LOOKUP\", \"WEATHER_LOOKUP\", \"FINANCIAL_DATA\", or \"NO_LOOKUP\".";

                    let decider_messages = vec![ChatMessage {
                        role: "user".to_string(),
                        content: decider_prompt,
                    }];
                    let decider_model_name = "gemini-2.0-flash".to_string();

                    let decider_gemini_api_key_string = match config.gemini_api_key.clone() {
                        Some(key) if !key.is_empty() => key,
                        _ => {
                            log::warn!(
                                "Gemini API key not set for decider. Defaulting to NO_LOOKUP."
                            );
                            String::new()
                        }
                    };

                    let decision: String; // Initialize decision
                    if !decider_gemini_api_key_string.is_empty() {
                        match call_gemini_api_non_streaming(
                            &client,
                            decider_messages,
                            &decider_gemini_api_key_string,
                            decider_model_name.clone(),
                        )
                        .await
                        {
                            Ok(decider_response_text) => {
                                let cleaned_response = decider_response_text.trim().to_uppercase();
                                log::info!(
                                    "Decider model response for query '{}': '{}'",
                                    user_query,
                                    cleaned_response
                                );
                                if [
                                    "WIKIPEDIA_LOOKUP",
                                    "WEATHER_LOOKUP",
                                    "FINANCIAL_DATA",
                                    "NO_LOOKUP",
                                ]
                                .contains(&cleaned_response.as_str())
                                {
                                    decision = cleaned_response;
                                } else {
                                    log::warn!("Decider model returned an unexpected response: '{}'. Defaulting to NO_LOOKUP.", decider_response_text);
                                    decision = "NO_LOOKUP".to_string();
                                }
                            }
                            Err(e) => {
                                log::error!("Error calling decider model for query '{}': {}. Defaulting to NO_LOOKUP.", user_query, e);
                                decision = "NO_LOOKUP".to_string();
                            }
                        }
                    } else {
                        log::warn!("Decider Gemini API key is empty. Defaulting to NO_LOOKUP for query '{}'.", user_query);
                        decision = "NO_LOOKUP".to_string(); // Ensure decision is NO_LOOKUP if key is empty
                    }

                    // Replace if decision == "WEB_SEARCH" logic with new cases
                    if decision == "WIKIPEDIA_LOOKUP" {
                        log::info!(
                            "Iterative Wikipedia lookup DECIDED for query: '{}'",
                            user_query
                        );
                        let max_iterations = 4; // Max iterations for the research

                        if let Err(e) = window.emit(
                            "ARTICLE_LOOKUP_STARTED",
                            ArticleLookupStartedPayload {
                                query: user_query.to_string(), // Use the original user query for the event
                            },
                        ) {
                            log::warn!("Failed to emit ARTICLE_LOOKUP_STARTED event: {}", e);
                        }

                        match perform_iterative_wikipedia_research(
                            &client,
                            user_query,
                            &decider_gemini_api_key_string, // API key
                            &decider_model_name,            // Model for internal calls
                            max_iterations,
                        )
                        .await
                        {
                            Ok(results) => {
                                if results.is_empty() {
                                    log::info!("Iterative Wikipedia lookup for query '{}' completed, but no specific information found.", user_query);
                                    if let Err(e) = window.emit(
                                                               "ARTICLE_LOOKUP_COMPLETED",
                                                               ArticleLookupCompletedPayload {
                                                                   query: user_query.to_string(),
                                                                   success: true, // Process completed
                                                                   summary: Some("No specific information found after iterative search.".to_string()),
                                                                   source_name: None,
                                                                   source_url: None,
                                                                   error: None,
                                                               },
                                                           ) {
                                                               log::warn!("Failed to emit ARTICLE_LOOKUP_COMPLETED (no results) event: {}", e);
                                                           }
                                } else {
                                    log::info!("Iterative Wikipedia lookup successful for query: '{}'. Found {} results.", user_query, results.len());
                                    let mut combined_summary = String::new();
                                    let mut combined_source_names = Vec::<String>::new();
                                    let mut combined_source_urls = Vec::<String>::new();

                                    for (_i, res) in results.iter().enumerate() {
                                        combined_summary.push_str(&format!(
                                            "Title: {}\nSummary: {}\n\n",
                                            res.title, res.summary,
                                        ));
                                        combined_source_names.push(res.title.clone());
                                        combined_source_urls.push(res.url.clone());
                                    }

                                    article_lookup_result_text = Some(format!(
                                                               "Context from Iterative Wikipedia Search for user query '{}':\n\n{}",
                                                               user_query,
                                                               combined_summary.trim_end()
                                                           ));
                                    article_lookup_performed_successfully = true;

                                    if let Err(e) = window.emit(
                                        "ARTICLE_LOOKUP_COMPLETED",
                                        ArticleLookupCompletedPayload {
                                            query: user_query.to_string(),
                                            success: true,
                                            summary: Some(combined_summary),
                                            source_name: Some(combined_source_names),
                                            source_url: Some(combined_source_urls),
                                            error: None,
                                        },
                                    ) {
                                        log::warn!("Failed to emit ARTICLE_LOOKUP_COMPLETED (success) event: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                log::error!(
                                    "Iterative Wikipedia lookup failed for query '{}'. Error: {}",
                                    user_query,
                                    e
                                );
                                if let Err(emit_err) = window.emit(
                                    "ARTICLE_LOOKUP_COMPLETED",
                                    ArticleLookupCompletedPayload {
                                        query: user_query.to_string(),
                                        success: false,
                                        summary: None,
                                        source_name: None,
                                        source_url: None,
                                        error: Some(e.clone()),
                                    },
                                ) {
                                    log::warn!(
                                        "Failed to emit ARTICLE_LOOKUP_COMPLETED (error) event: {}",
                                        emit_err
                                    );
                                }
                            }
                        }
                    } else if decision == "WEATHER_LOOKUP" {
                        log::info!("Weather lookup DECIDED for query: '{}'", user_query);

                        // Pass the original user_query to perform_weather_lookup,
                        // which will internally call the location extractor.
                        // Also pass the Gemini API key and a model for the extractor.
                        if let Err(e) = window.emit(
                            "WEATHER_LOOKUP_STARTED",
                            WeatherLookupStartedPayload {
                                location: user_query.to_string(),
                            },
                        ) {
                            log::warn!("Failed to emit WEATHER_LOOKUP_STARTED event: {}", e);
                        }
                        match perform_weather_lookup(
                            &client,
                            user_query,
                            &decider_gemini_api_key_string,
                            "gemini-2.0-flash".to_string(),
                        )
                        .await
                        {
                            Ok(Some((temp, unit, description, resolved_location))) => {
                                log::info!("Weather lookup successful for '{}' (resolved: {}). Temp: {} {}", user_query, resolved_location, temp, unit);
                                if let Err(e) = window.emit(
                                    "WEATHER_LOOKUP_COMPLETED",
                                    WeatherLookupCompletedPayload {
                                        location: resolved_location.clone(), // Use the (potentially more precise) resolved location from geocoding
                                        success: true,
                                        temperature: Some(temp),
                                        unit: Some(unit.clone()),
                                        description: Some(description.clone()),
                                        error: None,
                                    },
                                ) {
                                    log::warn!("Failed to emit WEATHER_LOOKUP_COMPLETED (success) event: {}", e);
                                }
                                weather_lookup_result_text = Some(format!(
                                    "Current weather for {}: {} {}. {}.",
                                    resolved_location, temp, unit, description
                                ));
                                weather_lookup_performed_successfully = true;
                            }
                            Ok(None) => {
                                log::info!("Weather lookup for '{}' completed, but no weather data found (likely geocoding or location extraction failed, or no data for coords).", user_query);
                                if let Err(e) = window.emit(
                                    "WEATHER_LOOKUP_COMPLETED",
                                    WeatherLookupCompletedPayload {
                                        location: user_query.to_string(), // Fallback to original query for event if resolution failed
                                        success: true,
                                        temperature: None,
                                        unit: None,
                                        description: None,
                                        error: None,
                                    },
                                ) {
                                    log::warn!("Failed to emit WEATHER_LOOKUP_COMPLETED (no data) event: {}", e);
                                }
                            }
                            Err(e) => {
                                log::error!(
                                    "Weather lookup failed for '{}'. Error: {}",
                                    user_query,
                                    e
                                );
                                if let Err(emit_err) = window.emit(
                                    "WEATHER_LOOKUP_COMPLETED",
                                    WeatherLookupCompletedPayload {
                                        location: user_query.to_string(),
                                        success: false,
                                        temperature: None,
                                        unit: None,
                                        description: None,
                                        error: Some(e.clone()),
                                    },
                                ) {
                                    log::warn!(
                                        "Failed to emit WEATHER_LOOKUP_COMPLETED (error) event: {}",
                                        emit_err
                                    );
                                }
                            }
                        }
                    } else if decision == "FINANCIAL_DATA" {
                        log::info!("Financial data lookup DECIDED for query: '{}'", user_query);
                        if let Some(symbol) = extract_stock_symbol(user_query) {
                            log::info!("Extracted symbol '{}' for financial data lookup.", symbol);
                            if let Err(e) = window.emit(
                                "FINANCIAL_DATA_STARTED",
                                FinancialDataStartedPayload {
                                    query: user_query.to_string(),
                                    symbol: symbol.clone(),
                                },
                            ) {
                                log::warn!("Failed to emit FINANCIAL_DATA_STARTED event: {}", e);
                            }

                            match perform_financial_data_lookup(&client, &symbol).await {
                                Ok(data) => {
                                    log::info!(
                                        "Financial data lookup successful for symbol: '{}'.",
                                        symbol
                                    );
                                    if let Err(e) = window.emit(
                                        "FINANCIAL_DATA_COMPLETED",
                                        FinancialDataCompletedPayload {
                                            query: user_query.to_string(),
                                            symbol: symbol.clone(),
                                            success: true,
                                            data: Some(data.clone()),
                                            error: None,
                                        },
                                    ) {
                                        log::warn!("Failed to emit FINANCIAL_DATA_COMPLETED (success) event: {}", e);
                                    }
                                    financial_data_result_text =
                                        Some(format!("Financial data for {}\n{}", symbol, data));
                                    financial_data_fetched_successfully = true;
                                }
                                Err(e) => {
                                    log::error!(
                                        "Financial data lookup failed for symbol: '{}'. Error: {}",
                                        symbol,
                                        e
                                    );
                                    if let Err(emit_err) = window.emit(
                                        "FINANCIAL_DATA_COMPLETED",
                                        FinancialDataCompletedPayload {
                                            query: user_query.to_string(),
                                            symbol: symbol.clone(),
                                            success: false,
                                            data: None,
                                            error: Some(e.clone()),
                                        },
                                    ) {
                                        log::warn!("Failed to emit FINANCIAL_DATA_COMPLETED (error) event: {}", emit_err);
                                    }
                                }
                            }
                        } else {
                            log::warn!("Financial data lookup decided, but could not extract symbol from query: '{}'. Skipping financial lookup.", user_query);
                            // Emit a FINANCIAL_DATA_COMPLETED event to inform the frontend about the symbol extraction failure.
                            if let Err(e) = window.emit(
                                "FINANCIAL_DATA_COMPLETED",
                                FinancialDataCompletedPayload {
                                    query: user_query.to_string(),
                                    symbol: user_query.to_string(), // Use original query as a fallback for display
                                    success: false,
                                    data: None,
                                    error: Some(
                                        "Could not identify a stock symbol in your query"
                                            .to_string(),
                                    ),
                                },
                            ) {
                                log::warn!("Failed to emit FINANCIAL_DATA_COMPLETED (symbol extraction failure) event: {}", e);
                            }
                        }
                    } else {
                        // NO_LOOKUP
                        log::info!(
                            "External data lookup (Web/Financial) NOT decided for query: '{}'",
                            user_query
                        );
                    }
                }
            }
        }
    }

    // Construct final message list for LLM
    if article_lookup_performed_successfully && article_lookup_result_text.is_some() {
        final_messages.push(ChatMessage {
            role: "user".to_string(),
            content: format!(
                "Context from Wikipedia lookup:\n{}\n\nGiven this context, please answer the following user query:",
                article_lookup_result_text.unwrap()
            ),
        });
    } else if weather_lookup_performed_successfully && weather_lookup_result_text.is_some() {
        final_messages.push(ChatMessage {
            role: "user".to_string(),
            content: format!(
                "Context from Weather lookup:\n{}\n\nGiven this context, please answer the following user query:",
                weather_lookup_result_text.unwrap()
            ),
        });
    } else if financial_data_fetched_successfully && financial_data_result_text.is_some() {
        final_messages.push(ChatMessage {
            role: "user".to_string(),
            content: format!(
                "Context from Financial data lookup:\n{}\n\nGiven this context, please answer the following user query:",
                financial_data_result_text.unwrap()
            ),
        });
    }

    // Append original user messages
    final_messages.extend(messages.into_iter());

    // Check if the model is a Gemini model
    if model_name.starts_with("gemini-") || model_name.starts_with("google/") {
        // Crude check, refine as needed
        let gemini_api_key = match config.gemini_api_key {
            Some(key) if !key.is_empty() => key,
            _ => {
                log::error!(
                    "Gemini API key is not set in config for model: {}",
                    model_name
                );
                return Err(
                    "Gemini API key is not configured. Please set it in settings.".to_string(),
                );
            }
        };
        log::info!("Using Gemini API for model: {}", model_name);

        match call_gemini_api(
            &client, // Pass client
            final_messages,
            gemini_api_key,
            model_name.replace("google/", ""),
            window.clone(),
        )
        .await
        {
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
                log::error!(
                    "OpenRouter API key is not set in config for model: {}",
                    model_name
                );
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
        match call_openrouter_api(&client, final_messages, api_key, model_name, window.clone())
            .await
        {
            // Pass client
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
        "gemini-2.0-flash", // Keep this for potential direct use or alias
        "gemini-2.5-flash-preview-05-20", // This is the "Gemini 2.5 Flash (non-thinking)"
        "gemini-2.5-flash-preview-05-20#thinking-enabled",
    ];
    // Updated check to be more specific
    if !allowed_models.contains(&model_name.as_str()) {
        log::error!("Attempted to set invalid model: {}", model_name);
        return Err(format!(
            "Invalid model selection: {}. Allowed models are: {:?}",
            model_name, allowed_models
        ));
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

// --- ADDED: Command to set web search preference ---
#[tauri::command]
async fn set_enable_web_search(enable: bool, app_handle: AppHandle) -> Result<(), String> {
    let mut config = load_config(&app_handle).unwrap_or_else(|e| {
        log::warn!(
            "Failed to load config when setting web search preference: {}. Using default.",
            e
        );
        AppConfig::default()
    });
    config.enable_web_search = Some(enable);
    save_config(&app_handle, &config)
}

// --- ADDED: Command to get web search preference ---
#[tauri::command]
async fn get_enable_web_search(app_handle: AppHandle) -> Result<bool, String> {
    load_config(&app_handle).map(|config| config.enable_web_search.unwrap_or(true))
    // Default to true
}

// --- API Call Logic ---
async fn call_gemini_api(
    client: &reqwest::Client, // MODIFIED: Accept client
    messages: Vec<ChatMessage>,
    api_key: String,
    model_identifier_from_config: String, // RENAMED for clarity
    window: Window,
) -> Result<(), String> {
    let mut actual_model_name_for_api = model_identifier_from_config.clone();
    let mut gen_config: Option<GenerationConfigForGemini> = None;

    if model_identifier_from_config == "gemini-2.5-flash-preview-05-20" {
        // This is the "Gemini 2.5 Flash" (non-thinking explicit budget 0)
        gen_config = Some(GenerationConfigForGemini {
            thinking_config: Some(ThinkingConfig {
                include_thoughts: None, // Let API decide default or if it's implied by budget
                thinking_budget: Some(0),
            }),
            // ..Default::default() // for other potential future fields in GenerationConfigForGemini
        });
        // actual_model_name_for_api is already correct
    } else if model_identifier_from_config == "gemini-2.5-flash-preview-05-20#thinking-enabled" {
        // This is "Gemini 2.5 Flash (Thinking)" (default thinking, no specific budget)
        actual_model_name_for_api = "gemini-2.5-flash-preview-05-20".to_string(); // Use base model name for API
        gen_config = Some(GenerationConfigForGemini {
            thinking_config: Some(ThinkingConfig {
                include_thoughts: Some(true),
                thinking_budget: None,
            }),
            // This means include_thoughts is true and thinking_budget is non-zero.
        });
    }
    // For other gemini models, gen_config remains None (no other thinking models), and no specific generation_config will be sent.

    let api_url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?key={}&alt=sse",
        actual_model_name_for_api, // Use the potentially modified model name
        api_key
    );

    let request_payload = GeminiChatCompletionRequest {
        contents: messages
            .into_iter()
            .map(|msg| {
                let role_for_gemini = if msg.role == "assistant" {
                    "model".to_string()
                } else if msg.role == "system" {
                    // Our prepended system instruction
                    "user".to_string() // Gemini handles system prompts as initial "user" messages
                } else {
                    // "user" (from human actual input)
                    msg.role // Assuming it's "user"
                };
                GeminiContent {
                    parts: vec![GeminiPart { text: msg.content }],
                    role: Some(role_for_gemini),
                }
            })
            .collect(),
        generation_config: gen_config, // Set the generation_config
    };

    log::info!(
        "Sending STREAMING request to Gemini API for model: {} (API model: {}). Payload: {:?}",
        model_identifier_from_config,
        actual_model_name_for_api,
        request_payload
    );

    let response_result = client
        .post(&api_url)
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
                                        let line = line_buffer
                                            .drain(..newline_pos + 1)
                                            .collect::<String>();
                                        let trimmed_line = line.trim();

                                        if trimmed_line.starts_with("data: ") {
                                            let data_json_str = &trimmed_line[6..]; // Skip "data: "
                                                                                    // Gemini stream might send an array of responses, often with one element.
                                                                                    // And sometimes it sends a single JSON object directly.
                                                                                    // We need to handle both cases.
                                                                                    // The API doc (and community post) suggests each SSE event is one JSON object representing a GeminiChatCompletionResponse.

                                            // Attempt to parse as a single GeminiChatCompletionResponse
                                            match serde_json::from_str::<GeminiChatCompletionResponse>(
                                                data_json_str,
                                            ) {
                                                Ok(gemini_response_chunk) => {
                                                    let current_chunk_content: String;
                                                    let mut current_chunk_reasoning: Option<
                                                        String,
                                                    > = None;

                                                    // Process candidates for content
                                                    if let Some(candidate) =
                                                        gemini_response_chunk.candidates.get(0)
                                                    {
                                                        if let Some(part) =
                                                            candidate.content.parts.get(0)
                                                        {
                                                            let content_text = &part.text;

                                                            // Parse reasoning from content
                                                            let (content, reasoning) =
                                                                separate_reasoning_from_content(
                                                                    content_text,
                                                                );
                                                            current_chunk_content = content;
                                                            if !reasoning.is_empty() {
                                                                current_chunk_reasoning =
                                                                    Some(reasoning);
                                                            }

                                                            accumulated_content
                                                                .push_str(&current_chunk_content);

                                                            // Emit using new StreamChoiceDelta structure
                                                            if let Err(e) = window.emit(
                                                                "STREAM_CHUNK",
                                                                StreamChoiceDelta {
                                                                    content: if current_chunk_content.is_empty() { None } else { Some(current_chunk_content) },
                                                                    role: Some("assistant".to_string()),
                                                                    reasoning: current_chunk_reasoning,
                                                                },
                                                            ) {
                                                                log::error!("Failed to emit STREAM_CHUNK for Gemini: {}", e);
                                                            }
                                                        }
                                                    }
                                                }
                                                Err(e) => {
                                                    // It might be an array of these objects, though less common for pure SSE streams.
                                                    // The official docs for streamGenerateContent show each event as *one* GenerateContentResponse.
                                                    // So, if direct parsing fails, it's likely an error or an unexpected format.
                                                    if !data_json_str.is_empty()
                                                        && data_json_str != "["
                                                        && data_json_str != "]"
                                                    {
                                                        // Avoid logging for simple array brackets if they appear alone.
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
                                            log::warn!(
                                                "Unexpected line in Gemini stream: {}",
                                                trimmed_line
                                            );
                                        }
                                    }
                                }
                                Err(e) => {
                                    log::error!("Gemini stream chunk not valid UTF-8: {}", e);
                                    let _ = window.emit(
                                        "STREAM_ERROR",
                                        StreamErrorPayload {
                                            error: format!(
                                                "Gemini stream chunk not valid UTF-8: {}",
                                                e
                                            ),
                                        },
                                    );
                                    return Err(format!(
                                        "Gemini stream chunk not valid UTF-8: {}",
                                        e
                                    ));
                                }
                            }
                        }
                        Err(e) => {
                            log::error!("Error receiving stream chunk from Gemini: {}", e);
                            let _ = window.emit(
                                "STREAM_ERROR",
                                StreamErrorPayload {
                                    error: format!("Error in Gemini stream: {}", e),
                                },
                            );
                            return Err(format!("Error receiving Gemini stream chunk: {}", e));
                        }
                    }
                }
                // Stream ended
                log::info!(
                    "Gemini stream finished. Accumulated content: {}",
                    accumulated_content
                );

                // Final separation of reasoning from content for stream end
                let (final_content, final_reasoning) =
                    separate_reasoning_from_content(&accumulated_content);

                let _ = window.emit(
                    "STREAM_END",
                    StreamEndPayload {
                        full_content: final_content,
                        reasoning: if final_reasoning.is_empty() {
                            None
                        } else {
                            Some(final_reasoning)
                        },
                    },
                );
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
                let err_msg = format!(
                    "Gemini API (streaming) request failed: {} - {}",
                    status, error_text
                );
                let _ = window.emit(
                    "STREAM_ERROR",
                    StreamErrorPayload {
                        error: err_msg.clone(),
                    },
                );
                Err(err_msg)
            }
        }
        Err(e) => {
            log::error!("Network request to Gemini API (streaming) failed: {}", e);
            let err_msg = format!("Gemini API (streaming) network request failed: {}", e);
            let _ = window.emit(
                "STREAM_ERROR",
                StreamErrorPayload {
                    error: err_msg.clone(),
                },
            );
            Err(err_msg)
        }
    }
}

async fn call_openrouter_api(
    client: &reqwest::Client, // MODIFIED: Accept client
    messages: Vec<ChatMessage>,
    api_key: String,
    model_name: String,
    window: Window,
) -> Result<(), String> {
    let api_url = "https://openrouter.ai/api/v1/chat/completions";
    let mut request_payload = ChatCompletionRequest {
        model: model_name.clone(),
        messages: messages.clone(),
        stream: Some(true),
        include_reasoning: None,
    };

    // Enable reasoning for DeepSeek R1 models
    if model_name.starts_with("deepseek/deepseek-r1") {
        log::info!(
            "Enabling 'include_reasoning' for DeepSeek R1 model: {}",
            model_name
        );
        request_payload.include_reasoning = Some(true);
    }

    log::info!(
        "Sending streaming request to OpenRouter for model: {}. Payload: {:?}",
        model_name,
        request_payload
    );

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
                let mut accumulated_reasoning = String::new();
                let mut line_buffer = String::new();

                while let Some(item) = stream.next().await {
                    match item {
                        Ok(chunk_bytes) => {
                            match std::str::from_utf8(&chunk_bytes) {
                                Ok(chunk_str) => {
                                    line_buffer.push_str(chunk_str);

                                    // Process complete lines from the buffer
                                    while let Some(newline_pos) = line_buffer.find("\n") {
                                        let line = line_buffer
                                            .drain(..newline_pos + 1)
                                            .collect::<String>();
                                        let trimmed_line = line.trim();

                                        if trimmed_line.starts_with("data: ") {
                                            let data_json_str = &trimmed_line[6..];
                                            if data_json_str == "[DONE]" {
                                                log::info!("OpenRouter stream [DONE] received.");
                                                let final_reasoning =
                                                    if accumulated_reasoning.is_empty() {
                                                        None
                                                    } else {
                                                        Some(accumulated_reasoning)
                                                    };
                                                let _ = window.emit(
                                                    "STREAM_END",
                                                    StreamEndPayload {
                                                        full_content: accumulated_content.clone(),
                                                        reasoning: final_reasoning,
                                                    },
                                                );
                                                return Ok(()); // Successfully finished streaming
                                            }
                                            match serde_json::from_str::<
                                                StreamingChatCompletionResponse,
                                            >(
                                                data_json_str
                                            ) {
                                                Ok(parsed_chunk) => {
                                                    if let Some(choice) =
                                                        parsed_chunk.choices.get(0)
                                                    {
                                                        if let Some(content_delta) =
                                                            &choice.delta.content
                                                        {
                                                            accumulated_content
                                                                .push_str(content_delta);
                                                            if let Err(e) = window.emit(
                                                                "STREAM_CHUNK",
                                                                StreamChunkPayload {
                                                                    delta: Some(
                                                                        content_delta.clone(),
                                                                    ),
                                                                },
                                                            ) {
                                                                log::error!("Failed to emit STREAM_CHUNK: {}", e);
                                                            }
                                                        }
                                                        // Capture reasoning delta if present
                                                        if let Some(reasoning_delta) =
                                                            &choice.delta.reasoning
                                                        {
                                                            if !reasoning_delta.is_empty() {
                                                                log::debug!("Received reasoning delta for OpenRouter: '{}'", reasoning_delta);
                                                                accumulated_reasoning
                                                                    .push_str(reasoning_delta);
                                                            }
                                                        }
                                                    }
                                                }
                                                Err(e) => {
                                                    // Ignore lines that are not valid JSON data chunks, could be comments or empty lines
                                                    if !data_json_str.is_empty()
                                                        && !data_json_str.starts_with(":")
                                                    {
                                                        log::warn!("Failed to parse stream data JSON from OpenRouter: '{}'. Raw: '{}'", e, data_json_str);
                                                    }
                                                }
                                            }
                                        } else if !trimmed_line.is_empty()
                                            && !trimmed_line.starts_with(":")
                                        {
                                            // Log unexpected non-empty, non-comment lines
                                            log::warn!(
                                                "Unexpected line in OpenRouter stream: {}",
                                                trimmed_line
                                            );
                                        }
                                    }
                                }
                                Err(e) => {
                                    log::error!("Stream chunk not valid UTF-8: {}", e);
                                    let _ = window.emit(
                                        "STREAM_ERROR",
                                        StreamErrorPayload {
                                            error: format!("Stream chunk not valid UTF-8: {}", e),
                                        },
                                    );
                                    return Err(format!("Stream chunk not valid UTF-8: {}", e));
                                }
                            }
                        }
                        Err(e) => {
                            log::error!("Error receiving stream chunk from OpenRouter: {}", e);
                            let _ = window.emit(
                                "STREAM_ERROR",
                                StreamErrorPayload {
                                    error: format!("Error in stream: {}", e),
                                },
                            );
                            return Err(format!("Error receiving stream chunk: {}", e));
                        }
                    }
                }
                // If loop finishes without [DONE], it might be an incomplete stream or an issue.
                // Emit an error or handle as appropriate. For now, assume [DONE] is the primary exit.
                log::warn!("OpenRouter stream ended without [DONE] marker.");
                let _ = window.emit(
                    "STREAM_ERROR",
                    StreamErrorPayload {
                        error: "Stream ended without [DONE] marker".to_string(),
                    },
                );
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
                let _ = window.emit(
                    "STREAM_ERROR",
                    StreamErrorPayload {
                        error: err_msg.clone(),
                    },
                );
                Err(err_msg)
            }
        }
        Err(e) => {
            log::error!("Network request to OpenRouter failed: {}", e);
            let err_msg = format!("Network request failed: {}", e);
            let _ = window.emit(
                "STREAM_ERROR",
                StreamErrorPayload {
                    error: err_msg.clone(),
                },
            );
            Err(err_msg)
        }
    }
}

// --- ADDED: Non-streaming Gemini API call function ---
async fn call_gemini_api_non_streaming(
    client: &reqwest::Client,
    messages: Vec<ChatMessage>,
    api_key_slice: &str, // Changed parameter name for clarity
    model_name: String,
) -> Result<String, String> {
    if api_key_slice.is_empty() {
        return Err("API key is empty for non-streaming Gemini call".to_string());
    }
    let api_url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model_name, api_key_slice
    );

    // For a simple YES/NO decider, complex generation_config is not needed.
    // We can omit it or send a minimal one if required by the API.
    // For now, omitting `generation_config` for simplicity for the decider call.
    let request_payload = GeminiChatCompletionRequest {
        contents: messages
            .into_iter()
            .map(|msg| GeminiContent {
                parts: vec![GeminiPart { text: msg.content }],
                role: Some(msg.role), // Directly use the role, assuming "user" for decider prompt
            })
            .collect(),
        generation_config: None, // No special generation config for the simple decider
    };

    // log::info!(
    //     "Sending NON-STREAMING request to Gemini API for model: {}. Payload: {:?}",
    //     model_name,
    //     request_payload
    // );

    match client
        .post(&api_url)
        .header("Content-Type", "application/json")
        .json(&request_payload)
        .send()
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                match response.json::<GeminiChatCompletionResponse>().await {
                    Ok(gemini_response) => {
                        if let Some(candidate) = gemini_response.candidates.get(0) {
                            if let Some(part) = candidate.content.parts.get(0) {
                                log::debug!("Non-streaming Gemini response text: {}", part.text);
                                Ok(part.text.clone())
                            } else {
                                Err("Non-streaming Gemini response: No content parts found"
                                    .to_string())
                            }
                        } else {
                            Err("Non-streaming Gemini response: No candidates found".to_string())
                        }
                    }
                    Err(e) => Err(format!(
                        "Failed to parse non-streaming Gemini JSON response: {}",
                        e
                    )),
                }
            } else {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_else(|_| {
                    "Could not read error body from Gemini (non-streaming)".to_string()
                });
                log::error!(
                    "Gemini API (non-streaming) request failed with status {}: {}",
                    status,
                    error_text
                );
                Err(format!(
                    "Gemini API (non-streaming) request failed: {} - {}",
                    status, error_text
                ))
            }
        }
        Err(e) => {
            log::error!(
                "Network request to Gemini API (non-streaming) failed: {}",
                e
            );
            Err(format!(
                "Gemini API (non-streaming) network request failed: {}",
                e
            ))
        }
    }
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn window_should_become_key(_panel: Panel) -> bool {
    log::info!("NSPanelDelegate: windowShouldBecomeKey called, returning false to prevent focus.");
    false
}

// --- ADDED: Location Extractor Function for Geocoding ---
async fn extract_location_for_geocoding(
    client: &reqwest::Client,
    user_query: &str, // The full user query, e.g., "what is the weather in Paris, France?"
    gemini_api_key: &str, // API key as a slice
    model_name: String, // Model name for Gemini
) -> Result<String, String> {
    // Returns the extracted location string or an error
    let extractor_prompt = format!(
        "{}{}{}{}{}{}{}{}{}{}",
        "You are an expert at identifying the geographical location mentioned in a user\'s query about weather.\n",
        "Given the user query, extract only the location (city, state, country, etc.). Do not include phrases like \"weather in\", \"what is the temperature in\", etc.\n",
        "For example:\n",
        "- User Query: \"weather in San Francisco, CA\" -> Location: \"San Francisco, CA\"\n",
        "- User Query: \"what is the temperature in London today?\" -> Location: \"London\"\n",
        "- User Query: \"Is it raining in Tokyo, Japan? Show me the forecast.\" -> Location: \"Tokyo, Japan\"\n",
        "- User Query: \"Paris forecast\" -> Location: \"Paris\"\n",
        "Output only the location itself.\n\n",
        format!("User Query: '{}'\n", user_query),
        "Location:"
    );

    let extractor_messages = vec![ChatMessage {
        role: "user".to_string(),
        content: extractor_prompt,
    }];

    log::info!(
        "Requesting location extraction for geocoding from query: '{}'",
        user_query
    );

    match call_gemini_api_non_streaming(client, extractor_messages, gemini_api_key, model_name)
        .await
    {
        Ok(extracted_location_raw) => {
            let extracted_location = extracted_location_raw.trim().trim_matches('"').to_string();
            log::info!(
                "Extracted location for geocoding: '{}' from original query: '{}'",
                extracted_location,
                user_query
            );
            if extracted_location.is_empty() {
                log::warn!("Location extractor for geocoding returned empty. Falling back to original query (trimmed).");
                Ok(user_query.trim().to_string()) // Fallback, though less ideal
            } else {
                Ok(extracted_location)
            }
        }
        Err(e) => {
            log::error!("Error calling location extractor for geocoding (query: '{}'): {}. Falling back to original query (trimmed).", user_query, e);
            Ok(user_query.trim().to_string()) // Fallback on error
        }
    }
}

// --- ADDED: Wikipedia Search Term Extractor Function ---
async fn extract_wikipedia_search_term(
    client: &reqwest::Client,
    user_query: &str,
    gemini_api_key_string: String,
    model_name: String,
) -> Result<Vec<String>, String> {
    let extractor_prompt = format!(
        "You are an expert at identifying core subjects or named entities in a user's query that are suitable for Wikipedia searches.\n\
        Given the user query, extract a list of concise and accurate search terms for Wikipedia.\n\
        If the query is simple, provide a single search term in the list.\n\
        If the query is complex or multifaceted, break it down into multiple relevant search terms.\n\
        Focus on the main topics, persons, places, or concepts.\n\
        Do not include conversational phrases like 'tell me about', 'what is', 'who was'.\n\
        Output the search terms as a JSON array of strings. For example: [\"Term 1\", \"Term 2\"]. If only one term, output as [\"Term\"].\n\n\
        Examples:\n\
        - User Query: \"Tell me more about the history of the Eiffel Tower in Paris.\"\n\
          Search Terms (JSON): [\"Eiffel Tower\"]\n\
        - User Query: \"Who was the first president of the United States?\"\n\
          Search Terms (JSON): [\"George Washington\"]\n\
        - User Query: \"What are the symptoms of influenza?\"\n\
          Search Terms (JSON): [\"Influenza\"]\n\
        - User Query: \"Explain quantum entanglement for me.\"\n\
          Search Terms (JSON): [\"Quantum entanglement\"]\n\
        - User Query: \"Compare the economies of Germany and France.\"\n\
          Search Terms (JSON): [\"Economy of Germany\", \"Economy of France\"]\n\
        - User Query: \"Compare Federer and Nadal.\"\n\
          Search Terms (JSON): [\"Roger Federer\", \"Rafael Nadal\"]\n\
        - User Query: \"Impact of Renaissance art on Baroque architecture.\"\n\
          Search Terms (JSON): [\"Renaissance art\", \"Baroque architecture\"]\n\
        User Query: '{}'\n\
        Search Terms (JSON):",
        user_query
    );

    let extractor_messages = vec![ChatMessage {
        role: "user".to_string(),
        content: extractor_prompt,
    }];

    log::info!(
        "Requesting Wikipedia search term extraction for query: '{}'",
        user_query
    );

    match call_gemini_api_non_streaming(
        client,
        extractor_messages,
        &gemini_api_key_string,
        model_name,
    )
    .await
    {
        Ok(response_str) => match serde_json::from_str::<Vec<String>>(&response_str) {
            Ok(terms) => {
                if terms.is_empty() {
                    log::warn!("Wikipedia search term extractor returned an empty list for query: '{}'. Falling back to original query.", user_query);
                    Ok(vec![user_query.to_string()])
                } else {
                    log::info!(
                        "Extracted Wikipedia search terms: {:?} for original query: '{}'",
                        terms,
                        user_query
                    );
                    Ok(terms)
                }
            }
            Err(e) => {
                log::error!("Failed to parse Wikipedia search terms from LLM response for query '{}'. Error: {}. Response: \"{}\". Falling back to original query.", user_query, e, response_str);
                Ok(vec![user_query.to_string()])
            }
        },
        Err(e) => {
            log::error!("Error calling Wikipedia search term extractor for query '{}': {}. Falling back to original query.", user_query, e);
            Ok(vec![user_query.to_string()])
        }
    }
}

async fn analyze_wikipedia_page_for_iteration(
    client: &reqwest::Client,
    original_user_query: &str,
    searched_term: &str,
    page_title: &str,
    page_content: &str,
    visited_page_titles: &[String],
    gemini_api_key: &str,
    model_name: &str,
) -> Result<AnalysisLLMDecision, String> {
    const MAX_CONTENT_CHARS: usize = 100000;
    let truncated_content = if page_content.chars().count() > MAX_CONTENT_CHARS {
        page_content
            .chars()
            .take(MAX_CONTENT_CHARS)
            .collect::<String>()
            + "\n[Content truncated]"
    } else {
        page_content.to_string()
    };

    let visited_titles_str = visited_page_titles.join(", ");

    let prompt = format!(
        "You are an AI assistant helping a user research a topic using Wikipedia. Your goal is to navigate Wikipedia pages iteratively to find the answer or relevant information for the user's original query.\n\n\
        Original User Query: \"{}\"\n\n\
        You have just read the Wikipedia page titled: \"{}\" (found by searching for \"{}\").\n\
        Here is the (potentially truncated) content of this page:\n---\n{}\n---\n\n\
        You have already visited or processed the following Wikipedia page titles in this research chain: [{}]. Do not suggest revisiting these.\n\n\
        Based on the original user query and the content of the current page, decide the next step:\n\
        1. If the current page's content directly and substantially answers the user's original query, or provides key information directly relevant to it: \
           Respond with a JSON object: {{\"decision_type\": \"FOUND_ANSWER\", \"summary\": \"<brief summary of the answer/info found on this page>\", \"title\": \"<current page title>\"}}\n\
        2. If the current page provides clues or mentions a more specific entity (person, place, event, concept, document, case name, etc.) that seems like a promising next step for a Wikipedia search to get closer to answering the original query: \
           Respond with a JSON object: {{\"decision_type\": \"NEXT_TERM\", \"term\": \"<concise Wikipedia search term for the next step>\", \"reason\": \"<briefly explain why this term is a good next step>\"}}. The term should be a precise Wikipedia article title if possible. Ensure the term is not in the list of already visited pages.\n\
        3. If the current page is not relevant, or doesn't offer a clear next step towards answering the query, or if you think the research path is a dead end: \
           Respond with a JSON object: {{\"decision_type\": \"STOP\", \"reason\": \"<briefly explain why you are stopping this path>\"}}\n\n\
        Focus on finding the most direct path to the answer. Be specific with \"NEXT_TERM\" suggestions. Ensure the JSON is valid.",
        original_user_query, page_title, searched_term, truncated_content, visited_titles_str
    );

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: prompt,
    }];

    log::info!(
        "Requesting Wikipedia content analysis for page: '{}', original query: '{}'",
        page_title,
        original_user_query
    );

    match call_gemini_api_non_streaming(client, messages, gemini_api_key, model_name.to_string())
        .await
    {
        Ok(response_str) => {
            log::debug!(
                "Raw analysis response for page '{}': {}",
                page_title,
                response_str
            );
            let cleaned_response = response_str
                .trim()
                .trim_start_matches("```json")
                .trim_start_matches("```")
                .trim_end_matches("```")
                .trim();
            match serde_json::from_str::<AnalysisLLMDecision>(cleaned_response) {
                Ok(decision) => Ok(decision),
                Err(e) => {
                    log::error!("Failed to parse analysis LLM response for page '{}'. Error: {}. Response: '{}', Cleaned: '{}'", page_title, e, response_str, cleaned_response);
                    Err(format!(
                        "Failed to parse analysis response: {}. Raw: {}",
                        e, response_str
                    ))
                }
            }
        }
        Err(e) => {
            log::error!(
                "Error calling analysis LLM for page '{}': {}",
                page_title,
                e
            );
            Err(format!("LLM call failed for analysis: {}", e))
        }
    }
}

pub async fn perform_iterative_wikipedia_research(
    client: &reqwest::Client,
    initial_user_query: &str,
    gemini_api_key: &str,
    model_name: &str,
    max_iterations: usize,
) -> Result<Vec<IterativeSearchResult>, String> {
    use std::collections::{HashSet, VecDeque};

    let mut all_found_info: Vec<IterativeSearchResult> = Vec::new();
    let mut visited_page_titles: HashSet<String> = HashSet::new();
    let mut search_queue: VecDeque<(String, Vec<String>)> = VecDeque::new();

    log::info!(
        "Starting iterative Wikipedia research for query: '{}'",
        initial_user_query
    );

    let initial_terms = match extract_wikipedia_search_term(
        client,
        initial_user_query,
        gemini_api_key.to_string(),
        model_name.to_string(),
    )
    .await
    {
        Ok(terms) => terms,
        Err(e) => {
            log::error!(
                "Failed initial term extraction for query '{}': {}",
                initial_user_query,
                e
            );
            // Fallback to using the original query if extraction fails
            vec![initial_user_query.to_string()]
        }
    };

    for term in initial_terms {
        if !term.trim().is_empty() {
            search_queue.push_back((term.clone(), vec![term]));
        }
    }

    if search_queue.is_empty() && !initial_user_query.trim().is_empty() {
        log::warn!("Initial term extraction yielded empty results for query: '{}'. Falling back to original query.", initial_user_query);
        search_queue.push_back((
            initial_user_query.to_string(),
            vec![initial_user_query.to_string()],
        ));
    }

    let mut current_iteration = 0;

    while let Some((current_term, current_path)) = search_queue.pop_front() {
        if current_iteration >= max_iterations {
            log::warn!(
                "Max iterations ({}) reached for query: {}",
                max_iterations,
                initial_user_query
            );
            break;
        }
        // Check based on the term we intend to search. Actual page titles are checked after lookup.
        if visited_page_titles.contains(&current_term) && current_path.len() > 1 {
            // Allow initial terms to be re-processed if they lead to different actual titles
            log::debug!(
                "Skipping already processed search term in path: {}",
                current_term
            );
            continue;
        }

        current_iteration += 1;
        log::info!(
            "Iterative search (iter {}/{}, path depth {}): Looking up '{}'. Path: {:?}",
            current_iteration,
            max_iterations,
            current_path.len(),
            current_term,
            current_path
        );

        match perform_wikipedia_lookup(client, &current_term).await {
            Ok(pages) => {
                let mut page_content_opt: Option<String> = None;
                let mut actual_page_title_opt: Option<String> = None;
                let mut page_url_opt: Option<String> = None;

                // The Wikipedia lookup returns a Vec of tuples (title, extract, url)
                for (title, extract, url) in pages {
                    if !extract.is_empty() {
                        page_content_opt = Some(extract.clone());
                        actual_page_title_opt = Some(title.clone());
                        page_url_opt = Some(url.clone());
                        break; // Found a usable page
                    }
                }

                if let (Some(content), Some(title), Some(url)) =
                    (page_content_opt, actual_page_title_opt, page_url_opt)
                {
                    if visited_page_titles.contains(&title) {
                        log::debug!("Skipping already visited Wikipedia page title: {}", title);
                        continue;
                    }

                    log::info!("Adding page to results: '{}'", title);
                    all_found_info.push(IterativeSearchResult {
                        title: title.clone(),
                        summary: content.clone(), // Using the full extract as the summary
                        url: url.clone(),
                        path_taken: current_path.clone(),
                    });

                    visited_page_titles.insert(title.clone());

                    // Only analyze if we haven't hit max_iterations for the *next* step
                    if current_iteration < max_iterations {
                        let visited_titles_vec: Vec<String> =
                            visited_page_titles.iter().cloned().collect();
                        match analyze_wikipedia_page_for_iteration(
                            client,
                            initial_user_query,
                            &current_term,
                            &title,
                            &content,
                            &visited_titles_vec,
                            gemini_api_key,
                            model_name,
                        )
                        .await
                        {
                            Ok(decision) => match decision {
                                AnalysisLLMDecision::FoundAnswer {
                                    summary: llm_summary,
                                    title: found_title,
                                } => {
                                    log::info!(
                                        "LLM indicated page '{}' (summary: '{}') as directly answering query '{}'. Information already captured.",
                                        found_title,
                                        llm_summary,
                                        initial_user_query
                                    );
                                    // Optionally, one could update the summary in all_found_info if llm_summary is preferred,
                                    // or simply stop this particular search path by not queueing further terms from it.
                                }
                                AnalysisLLMDecision::NextTerm {
                                    term: next_term,
                                    reason,
                                } => {
                                    log::info!(
                                        "Next term for '{}' is '{}'. Reason: {}",
                                        initial_user_query,
                                        next_term,
                                        reason
                                    );
                                    // Check conditions for adding to queue
                                    if !visited_page_titles.contains(&next_term)
                                        && !search_queue.iter().any(|(t, _)| t == &next_term)
                                        && current_path.len() < max_iterations
                                    // Path depth check
                                    {
                                        let mut next_path = current_path.clone();
                                        next_path.push(next_term.clone());
                                        search_queue.push_back((next_term, next_path));
                                    } else {
                                        log::debug!("Skipping next term suggestion '{}': already visited, in queue, or path too deep.", next_term);
                                    }
                                }
                                AnalysisLLMDecision::Stop { reason } => {
                                    log::info!(
                                        "Stopping search on path {:?} for query '{}'. Reason: {}",
                                        current_path,
                                        initial_user_query,
                                        reason
                                    );
                                }
                            },
                            Err(e) => {
                                log::error!("Error analyzing Wikipedia content for term '{}', page title '{}': {}", current_term, title, e);
                            }
                        }
                    } else {
                        log::info!("Max iterations reached after processing page '{}'. Not analyzing for next steps.", title);
                    }
                } else {
                    log::warn!(
                                            "No usable content found for Wikipedia term '{}' after processing API results.",
                                            current_term
                                        );
                    visited_page_titles.insert(current_term.clone()); // Mark term as processed to avoid retrying if it yields nothing
                }
            }
            Err(e) => {
                log::error!(
                    "Error performing Wikipedia lookup for term '{}': {}",
                    current_term,
                    e
                );
                visited_page_titles.insert(current_term.clone());
            }
        }
    }
    log::info!(
        "Finished iterative Wikipedia research for query: '{}'. Found {} results.",
        initial_user_query,
        all_found_info.len()
    );
    Ok(all_found_info)
}

// --- UPDATED: Weather Lookup Function (uses location extractor) ---
async fn perform_weather_lookup(
    client: &reqwest::Client,
    original_user_query: &str, // This is the full query like "weather in Paris"
    gemini_api_key_for_extractor: &str, // API key for the extractor LLM call
    extractor_model_name: String, // Model for the extractor LLM call
) -> Result<Option<(f32, String, String, String)>, String> {
    // (temp, unit, description, resolved_location)

    // 1. Extract location using the LLM extractor
    let location_to_geocode = match extract_location_for_geocoding(
        client,
        original_user_query,
        gemini_api_key_for_extractor,
        extractor_model_name,
    )
    .await
    {
        Ok(loc) => loc,
        Err(e) => {
            log::error!("Weather: Location extraction step failed for query '{}': {}. No geocoding will be attempted.", original_user_query, e);
            return Err(format!("Location extraction failed: {}", e)); // Propagate error if extraction itself fails badly
        }
    };

    // 2. Geocode the extracted location
    match geocode_location(client, &location_to_geocode).await {
        Ok(Some((lat, lon, resolved_geocoded_name))) => {
            log::info!(
                "Geocoded extracted location '{}' to ({}, {}), name: {}",
                location_to_geocode,
                lat,
                lon,
                resolved_geocoded_name
            );
            let base_url = "https://api.open-meteo.com/v1/forecast";
            let params = [
                ("latitude", lat.to_string()),
                ("longitude", lon.to_string()),
                ("current", "temperature_2m".to_string()),
                ("temperature_unit", "celsius".to_string()),
                ("wind_speed_unit", "kmh".to_string()),
                ("precipitation_unit", "mm".to_string()),
                ("timezone", "auto".to_string()),
            ];
            let request_url = client
                .get(base_url)
                .query(&params)
                .build()
                .unwrap()
                .url()
                .to_string();
            log::info!(
                "Weather lookup for ({}, {}). URL: {}",
                lat,
                lon,
                request_url
            );
            match client.get(base_url).query(&params).send().await {
                Ok(response) => {
                    let status = response.status();
                    let response_text = response
                        .text()
                        .await
                        .map_err(|e| format!("Weather: Failed to read response text: {}", e))?;
                    if status.is_success() {
                        match serde_json::from_str::<WeatherResponse>(&response_text) {
                            Ok(weather_data) => {
                                log::info!("Weather: Parsed JSON: {:#?}", weather_data);
                                if let Some(curr) = weather_data.current {
                                    if let (Some(temp_val), Some(units)) =
                                        (curr.temperature_2m, weather_data.current_units)
                                    {
                                        let unit = units
                                            .temperature_2m
                                            .unwrap_or_else(|| "°C".to_string());
                                        let desc = format!(
                                            "Current temperature in {}",
                                            resolved_geocoded_name
                                        );
                                        log::info!(
                                            "Weather: Found {} {} for {}",
                                            temp_val,
                                            unit,
                                            resolved_geocoded_name
                                        );
                                        return Ok(Some((
                                            temp_val,
                                            unit,
                                            desc,
                                            resolved_geocoded_name.clone(),
                                        ))); // No deref needed for f32
                                    }
                                }
                                log::info!("Weather: No current data for ({}, {}).", lat, lon);
                                Ok(None)
                            }
                            Err(e) => {
                                log::error!(
                                    "Weather: JSON parse error for ({}, {}): {}. Raw: {}",
                                    lat,
                                    lon,
                                    e,
                                    response_text
                                );
                                Err(format!(
                                    "Weather JSON error: {}. Ensure response is valid JSON.",
                                    e
                                ))
                            }
                        }
                    } else {
                        log::error!(
                            "Weather: API error for ({}, {}) status {}: {}",
                            lat,
                            lon,
                            status,
                            response_text
                        );
                        Err(format!("Weather API error: {} - {}", status, response_text))
                    }
                }
                Err(e) => {
                    log::error!("Weather: Network error for ({}, {}): {}", lat, lon, e);
                    Err(format!("Weather network error: {}", e))
                }
            }
        }
        Ok(None) => {
            log::warn!("Weather: Geocoding failed for '{}'.", location_to_geocode);
            Ok(None)
        }
        Err(e) => {
            log::error!(
                "Weather: Geocoding step failed for '{}': {}",
                location_to_geocode,
                e
            );
            Err(e)
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
    let show_hide_shortcut_definition =
        tauri_gs::Shortcut::new(Some(show_hide_modifiers), tauri_gs::Code::KeyZ);

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
        .plugin(tauri_nspanel::init())
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

            // Convert the main window to a panel (for macOS only)
            #[cfg(target_os = "macos")]
            {
                #[allow(non_upper_case_globals)]
                const NSWindowStyleMaskNonActivatingPanel: i32 = 1 << 7;
                if let Some(window) = app.get_webview_window("main") {
                    match window.to_panel() {
                        Ok(panel) => {
                            panel.set_released_when_closed(true);
                            log::info!("Successfully converted main window to NSPanel.");

                            // Set the style mask to make it a non-activating panel
                            panel.set_style_mask(NSWindowStyleMaskNonActivatingPanel);
                            log::info!("Set NSWindowStyleMaskNonActivatingPanel(1 << 7) on NSPanel.");

                            // The following macro may use deprecated cocoa::base::id and nil, but
                            // this is required by the tauri_nspanel API for now.
                            #[allow(deprecated)]
                            let delegate = panel_delegate!(NSPanelDelegateHook {
                                window_should_become_key
                            });
                            panel.set_delegate(delegate);
                            log::info!("NSPanel delegate set to prevent focus.");
                        }
                        Err(e) => {
                            log::error!("Failed to convert main window to NSPanel: {:?}", e);
                        }
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
            set_gemini_api_key,
            trigger_backend_window_toggle,
            set_enable_web_search,
            get_enable_web_search
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
