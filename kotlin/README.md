# Cactus for Kotlin

Cactus is a library that provides native LLM inference for Android applications written in Kotlin. This library wraps the Cactus C++ library, providing a convenient Kotlin API.

## Features

- Load and run LLM models on Android devices
- Text generation with full parameter control
- Chat completions with various templates
- Text embeddings
- Tokenization and detokenization
- LoRA adapter support
- Session management

## Installation

Add the dependency to your `build.gradle` file:

```gradle
implementation 'com.cactus:cactus-kotlin:1.0.0'
```

## Usage

```kotlin
import com.cactus.kotlin.Cactus
import com.cactus.kotlin.LlamaContext
import com.cactus.kotlin.models.CompletionParams
import com.cactus.kotlin.models.ContextParams
import kotlinx.coroutines.launch
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers

// Initialize context
val contextParams = ContextParams(
    model = "/path/to/model.gguf",
    nCtx = 2048,
    nThreads = 4
)

// Create a LlamaContext
val context = LlamaContext.create(contextParams)

// Generate completion
val completionParams = CompletionParams(
    prompt = "Hello, my name is",
    temperature = 0.7f,
    maxTokens = 100
)

CoroutineScope(Dispatchers.IO).launch {
    val completion = context.completion(completionParams)
    println(completion.text)
    
    // Clean up when done
    context.release()
}
```

## Requirements

- Android API 21+
- 64-bit architecture (arm64-v8a or x86_64)

## License

[Your license information here]
