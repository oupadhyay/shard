# Shard

A Tauri application for screen interaction with AI models.

## Completed Features

*   Chat interface with OpenRouter models (DeepSeek V3, Gemini Flash, DeepSeek R1)
*   API Key Management
*   Model Selection
*   **Screen Capture + OCR Analysis** (Requires Tesseract)
*   Markdown Rendering (0.3.0 and after)
*   LaTeX Rendering (0.3.0 and after)
*   Streaming Responses (0.3.1 and after)
*   Background Mode (0.3.2 and [MacOS only](https://developer.apple.com/documentation/appkit/nspanel))

### In Progress Features

*   Copy to Clipboard (Idea 0.3.2 💡)

## IMPORTANT: Prerequisites

For the **Screen Capture + OCR Analysis** feature to work, you **MUST** have Tesseract OCR installed on your system.

The application uses the `rust-tesseract` crate, which relies on a system installation of Tesseract.

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

**MIT License**

Copyright (c) [2025] [Ojasw Upadhyay]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
