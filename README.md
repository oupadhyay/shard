# Shard

A Tauri application for easy interaction with AI models.

## Features

*   Chat Interface with Streamed Responses
*   Free Models (DeepSeek V3, Gemini Flash, DeepSeek R1)
*   API Key Management (OpenRouter & Google)
*   **Screen Capture + OCR Analysis** (Requires Tesseract)
*   Markdown & LaTeX Rendering
*   Background Mode ([MacOS only](https://developer.apple.com/documentation/appkit/nspanel))
*   System Prompt
*   Reasoning Data for R1 & Reasoning Summaries for Gemini
*   Financial Data from Yahoo Finance
*   Weather Data from Open-Meteo
*   General Data from Wikipedia
*   Wikipedia Research Loop (up to 4 searches)

### In Progress Features

*   YT Video Summarization (Idea 0.5.X 💡)
*   Custom System Prompt (Idea 0.5.X 💡)

Feel free to add more ideas in the Issues or contribute to the project with a PR!

## IMPORTANT: Prerequisites

For the **Screen Capture + OCR Analysis** feature to work, you **MUST** have Tesseract OCR installed on your system. The application uses the `rust-tesseract` crate, which relies on a system installation of Tesseract.

### Installation Instructions

*   **macOS (using Homebrew):**
    ```bash
    brew install tesseract
    brew install tesseract-lang # Installs all language packs
    ```
*   **Windows:**
    *   Download an installer from the official Tesseract documentation or repositories like [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Ensure you select the desired language packs during installation.
    *   **Crucially**, you must add the Tesseract installation directory to your system's `PATH` environment variable so the application can find it.
*   **Linux (Debian/Ubuntu):**
    ```bash
    sudo apt update
    sudo apt install tesseract-ocr
    sudo apt install tesseract-ocr-eng # Or other languages like tesseract-ocr-all
    ```
*   **Linux (Fedora):**
    ```bash
    sudo dnf install tesseract
    sudo dnf install tesseract-langpack-eng # Or other languages
    ```

Please refer to the [official Tesseract documentation](https://tesseract-ocr.github.io/tessdoc/) for the most up-to-date installation instructions for your specific operating system.

## Development

To build and run the application in development mode:

1.  Clone the repository:
    ```bash
    git clone <repository_url> # Replace with actual URL after publishing
    cd shard
    ```
2.  Install Rust and Node.js/npm if you haven't already. Refer to the official [Rust](https://www.rust-lang.org/tools/install) and [Node.js](https://nodejs.org/) websites.
3.  Install Tauri prerequisites: Follow the guide on the [Tauri website](https://tauri.app/v1/guides/getting-started/prerequisites).
4.  Install project dependencies:
    ```bash
    cargo fetch # Fetches Rust dependencies
    # Frontend dependencies might be needed depending on the chosen frontend setup
    # e.g., npm install if using a Node.js-based frontend
    ```
5.  Run the development server:
    ```bash
    cargo tauri dev
    ```

To build the application for production:

```bash
cargo tauri build
```

This will generate installers/packages in the `src-tauri/target/release/bundle/` directory.

## License

This project is licensed under the MIT License.
