# Cactus Flutter

A Flutter plugin for running local LLM models on mobile devices. This is a Flutter port of the [Cactus React Native](https://github.com/cactus-compute/cactus) library.

## Features

- Run LLM models locally on iOS and Android devices
- Support for chat completions with various formats
- Streaming token generation
- Text embeddings
- Tool/function calling
- Grammar-based structured outputs
- LoRA adapter support
- Model downloading

## Installation

```yaml
dependencies:
  cactus_flutter: ^0.0.1
```

## Usage

### Basic Completion

```dart
import 'package:cactus_flutter/cactus_flutter.dart';

Future<void> main() async {
  // Initialize the LLM context
  final context = await initLlama(
    ContextParams(
      model: 'path/to/model.gguf',
      nCtx: 2048,
      nGpuLayers: 16, // Use GPU acceleration on supported devices
    ),
    onProgress: (progress) {
      print('Loading model: ${progress * 100}%');
    },
  );
  
  // Generate text
  final result = await context.completion(
    CompletionParams(
      prompt: 'How does a jet engine work?',
      temperature: 0.7,
      nPredict: 1024,
    ),
  );
  
  print(result.text);
  
  // Release resources when done
  await context.release();
}
```

### Chat Completion

```dart
import 'package:cactus_flutter/cactus_flutter.dart';

Future<void> main() async {
  // Initialize the model
  final context = await initLlama(
    ContextParams(
      model: 'path/to/model.gguf',
    ),
  );
  
  // Create a chat conversation
  final messages = [
    ChatMessage(role: 'system', content: 'You are a helpful assistant.'),
    ChatMessage(role: 'user', content: 'Tell me about Flutter'),
  ];
  
  // Generate a response
  final result = await context.completion(
    CompletionParams(
      messages: messages,
      temperature: 0.7,
    ),
  );
  
  print(result.content);
  
  // Release resources
  await context.release();
}
```

### Streaming Completion

```dart
import 'package:cactus_flutter/cactus_flutter.dart';

Future<void> main() async {
  final context = await initLlama(
    ContextParams(model: 'path/to/model.gguf'),
  );
  
  // Stream tokens as they're generated
  final result = await context.completion(
    CompletionParams(
      prompt: 'Write a poem about programming:',
    ),
    callback: (token) {
      // Print each token as it's generated
      print(token.token);
    },
  );
  
  await context.release();
}
```

### Model Download

```dart
import 'package:cactus_flutter/cactus_flutter.dart';

Future<void> main() async {
  // Download a model if it doesn't exist already
  final modelPath = await downloadModelIfNotExists(
    modelUrl: 'https://huggingface.co/model.gguf',
    onProgress: (progress) {
      print('Download progress: $progress%');
    },
  );
  
  // Use the downloaded model
  final context = await initLlama(
    ContextParams(model: modelPath),
  );
  
  // Use the model...
  
  await context.release();
}
```

### JSON Structured Output

```dart
import 'package:cactus_flutter/cactus_flutter.dart';

Future<void> main() async {
  final context = await initLlama(
    ContextParams(model: 'path/to/model.gguf'),
  );
  
  final result = await context.completion(
    CompletionParams(
      prompt: 'List three planets with their diameters',
      responseFormat: CompletionResponseFormat(
        type: 'json_object',
        schema: {
          'type': 'object',
          'properties': {
            'planets': {
              'type': 'array',
              'items': {
                'type': 'object',
                'properties': {
                  'name': {'type': 'string'},
                  'diameter': {'type': 'number'},
                  'unit': {'type': 'string'}
                },
                'required': ['name', 'diameter', 'unit']
              }
            }
          }
        },
      ),
    ),
  );
  
  // Parse the JSON result
  final jsonResult = jsonDecode(result.text);
  
  await context.release();
}
```

## Platform-specific Setup

### Android

Add the following to your app's `android/app/build.gradle`:

```gradle
android {
    // ...
    
    packagingOptions {
        pickFirst 'lib/x86/libc++_shared.so'
        pickFirst 'lib/x86_64/libc++_shared.so'
        pickFirst 'lib/armeabi-v7a/libc++_shared.so'
        pickFirst 'lib/arm64-v8a/libc++_shared.so'
    }
}
```

### iOS

Add the following to your app's `ios/Podfile`:

```ruby
post_install do |installer|
  installer.pods_project.targets.each do |target|
    target.build_configurations.each do |config|
      config.build_settings['ENABLE_BITCODE'] = 'NO'
    end
  end
end
```

## License

MIT 