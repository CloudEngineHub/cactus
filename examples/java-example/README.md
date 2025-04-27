# Cactus Java Example App

This is a simple Android application that demonstrates how to use the Cactus library to run LLMs on Android using Java.

## Features

- Load GGUF model files from device storage
- Generate text completions with adjustable parameters
- Stream tokens in real time
- Display generation statistics (tokens per second, etc.)
- Stop generation mid-stream

## Requirements

- Android device with 64-bit architecture (ARM64 or x86_64)
- Android 5.0 (API level 21) or higher
- Sufficient storage space for GGUF model files (models can range from ~100MB to several GB)
- Sufficient RAM to load the model (varies by model size)

## Setup Instructions

1. Clone the repository
2. Open the project in Android Studio
3. Build and run the app on your device
4. Download a compatible GGUF model file to your device
5. In the app, enter the full path to the model file
6. Click "Load Model" and wait for it to load
7. Enter a prompt and click "Generate" to start text generation

## Compatible Models

This example app works with GGUF format models, which are the standard format for llama.cpp. You can find compatible models on [Hugging Face](https://huggingface.co/) or other LLM repositories.

For testing, we recommend smaller models like:
- TinyLlama
- Phi-2
- Mistral-7B

## Implementation Details

The example demonstrates:
- How to load a model with progress tracking
- How to set up model parameters
- How to generate text completions
- How to handle token streaming
- How to properly manage model resources

Refer to the `MainActivity.java` file for the implementation details.

## License

This example is part of the Cactus project and is licensed under the same terms as the main project. 