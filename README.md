# Shard

A Tauri application for easy interaction with AI models.

**Note:** Option (⌥) + Space is the shortcut to toggle visibility of Shard, and Option (⌥) + K is the shortcut for OCR capture

## Features

*   Chat Interface with Streamed Responses
*   Free Models (DeepSeek V3, Gemini Flash, DeepSeek R1)
*   API Key Management (OpenRouter & Google)
*   Screen Capture + OCR Analysis (Built-in)
*   Markdown & LaTeX Rendering
*   Background Panel Mode ([MacOS only](https://developer.apple.com/documentation/appkit/nspanel))
*   System Prompt
*   Reasoning Data for R1 & Reasoning Summaries for Gemini
*   Financial Data from Yahoo Finance
*   Weather Data from Open-Meteo
*   General Data from Wikipedia
*   Wikipedia Research Loop (up to 4 searches)
*   Research Paper Data from ArXiv

Feel free to add more ideas in the Issues or contribute to the project with a PR!

## Prerequisites

Shard comes with built-in OCR capabilities and doesn't require any additional system installations. The application includes all necessary components for text recognition.

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
