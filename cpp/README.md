# Cactus C++

Cross-platform C++ library for running LLMs, VLMs, and TTS models locally on devices.

## Installation

### CMake Integration

```cmake
add_subdirectory(cactus/cpp)
target_link_libraries(your_target cactus_core_lib)
target_include_directories(your_target PRIVATE cactus/cpp)
```

**Requirements:**
- C++17+
- CMake 3.14+

## Quick Start

```cpp
#include "cactus.h"
#include <iostream>

int main() {
    cactus::cactus_context context;
    
    common_params params;
    params.model.path = "model.gguf";
    params.n_ctx = 2048;
    params.n_gpu_layers = 99;
    params.cpuparams.n_threads = 4;
    
    if (!context.loadModel(params)) return 1;
    
    context.params.prompt = "Hello!";
    context.params.n_predict = 100;
    
    context.initSampling();
    context.rewind();
    context.beginCompletion();
    context.loadPrompt();
    
    while (context.has_next_token && !context.is_interrupted) {
        auto token = context.doCompletion();
        if (token.tok == -1) break;
    }
    
    std::cout << context.generated_text << std::endl;
    return 0;
}
```

## Streaming Chat

```cpp
#include "cactus.h"
#include <iostream>
#include <string>

class ChatBot {
private:
    cactus::cactus_context context;
    
public:
    bool init(const std::string& model_path) {
        common_params params;
        params.model.path = model_path;
        params.n_ctx = 4096;
        params.n_batch = 512;
        params.n_gpu_layers = 99;
        params.cpuparams.n_threads = 4;
        
        params.sampling.temp = 0.7f;
        params.sampling.top_k = 40;
        params.sampling.top_p = 0.9f;
        params.sampling.penalty_repeat = 1.1f;
        
        return context.loadModel(params);
    }
    
    std::string chat(const std::string& user_input) {
        std::string messages = R"([{"role": "user", "content": ")" + user_input + R"("}])";
        std::string formatted = context.getFormattedChat(messages, "");
        
        context.params.prompt = formatted;
        context.params.n_predict = 256;
        
        context.initSampling();
        context.rewind();
        context.beginCompletion();
        context.loadPrompt();
        
        std::string current_text;
        while (context.has_next_token && !context.is_interrupted) {
            auto token = context.doCompletion();
            if (token.tok == -1) break;
            
            if (context.generated_text.length() > current_text.length()) {
                std::string new_token = context.generated_text.substr(current_text.length());
                std::cout << new_token << std::flush;
                current_text = context.generated_text;
            }
        }
        
        return context.generated_text;
    }
};

int main() {
    ChatBot bot;
    if (!bot.init("model.gguf")) return 1;
    
    std::string input;
    while (true) {
        std::cout << "\nYou: ";
        std::getline(std::cin, input);
        if (input == "quit") break;
        
        std::cout << "Bot: ";
        bot.chat(input);
        std::cout << std::endl;
    }
    
    return 0;
}
```

## Core APIs

### Context Management

```cpp
cactus::cactus_context context;

common_params params;
params.model.path = "model.gguf";
params.n_ctx = 4096;                    // Context size
params.n_batch = 512;                   // Batch size
params.n_gpu_layers = 99;               // GPU layers (0 = CPU only)
params.cpuparams.n_threads = 8;         // CPU threads

// Sampling parameters
params.sampling.temp = 0.7f;            // Temperature (0.1-2.0)
params.sampling.top_k = 40;             // Top-k sampling
params.sampling.top_p = 0.9f;           // Nucleus sampling
params.sampling.penalty_repeat = 1.1f;  // Repetition penalty

bool success = context.loadModel(params);
```

### Generation Process

```cpp
context.params.prompt = "Your prompt";
context.params.n_predict = 200;

context.initSampling();
context.rewind();
context.beginCompletion();
context.loadPrompt();

while (context.has_next_token && !context.is_interrupted) {
    auto token = context.doCompletion();
    if (token.tok == -1) break;
}

std::string result = context.generated_text;
```

### Chat Templates

```cpp
std::string messages = R"([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
])";

std::string formatted = context.getFormattedChat(messages, "");
context.params.prompt = formatted;
```

## Vision Models

### Setup VLM

```cpp
bool setupVision(cactus::cactus_context& context, 
                const std::string& model_path,
                const std::string& mmproj_path) {
    common_params params;
    params.model.path = model_path;
    params.n_ctx = 4096;
    params.n_gpu_layers = 99;
    
    if (!context.loadModel(params)) return false;
    return context.initMultimodal(mmproj_path, true);
}
```

### Image Analysis

```cpp
std::string analyzeImage(cactus::cactus_context& context,
                        const std::string& image_path,
                        const std::string& question) {
    std::string messages = R"([{"role": "user", "content": [
        {"type": "image"}, 
        {"type": "text", "text": ")" + question + R"("}
    ]}])";
    
    context.params.prompt = context.getFormattedChat(messages, "");
    context.params.n_predict = 200;
    context.params.sampling.temp = 0.3f;
    
    context.initSampling();
    context.rewind();
    context.beginCompletion();
    
    std::vector<std::string> media_paths = {image_path};
    context.loadPrompt(media_paths);
    
    while (context.has_next_token && !context.is_interrupted) {
        auto token = context.doCompletion();
        if (token.tok == -1) break;
    }
    
    return context.generated_text;
}
```

### Complete Vision Example

```cpp
class VisionBot {
private:
    cactus::cactus_context context;
    
public:
    bool init(const std::string& model_path, const std::string& mmproj_path) {
        common_params params;
        params.model.path = model_path;
        params.n_ctx = 4096;
        params.n_gpu_layers = 99;
        params.cpuparams.n_threads = 4;
        
        if (!context.loadModel(params)) return false;
        if (!context.initMultimodal(mmproj_path, true)) return false;
        
        return context.isMultimodalSupportVision();
    }
    
    std::string describe(const std::string& image_path) {
        return analyzeImage(context, image_path, "Describe this image.");
    }
    
    std::string answer(const std::string& image_path, const std::string& question) {
        return analyzeImage(context, image_path, question);
    }
};

int main() {
    VisionBot bot;
    if (!bot.init("vlm-model.gguf", "mmproj.gguf")) return 1;
    
    std::string description = bot.describe("image.jpg");
    std::cout << "Description: " << description << std::endl;
    
    std::string answer = bot.answer("image.jpg", "What objects are visible?");
    std::cout << "Objects: " << answer << std::endl;
    
    return 0;
}
```

## Text-to-Speech

### Setup TTS

```cpp
bool setupTTS(cactus::cactus_context& context,
              const std::string& model_path,
              const std::string& vocoder_path) {
    common_params params;
    params.model.path = model_path;
    params.n_ctx = 2048;
    params.n_gpu_layers = 99;
    
    if (!context.loadModel(params)) return false;
    return context.initVocoder(vocoder_path);
}
```

### Generate Speech

```cpp
std::vector<float> generateSpeech(cactus::cactus_context& context, 
                                 const std::string& text) {
    std::string formatted = context.getFormattedAudioCompletion("", text);
    context.params.prompt = formatted;
    context.params.n_predict = 500;
    
    auto guide_tokens = context.getAudioCompletionGuideTokens(text);
    context.setGuideTokens(guide_tokens);
    
    context.initSampling();
    context.rewind();
    context.beginCompletion();
    context.loadPrompt();
    
    std::vector<llama_token> audio_tokens;
    while (context.has_next_token && !context.is_interrupted) {
        auto token = context.doCompletion();
        if (token.tok >= 151672 && token.tok <= 155772) {
            audio_tokens.push_back(token.tok);
        }
        if (token.tok == 151668) break; // End token
    }
    
    return context.decodeAudioTokens(audio_tokens);
}
```

### Save as WAV

```cpp
void saveWAV(const std::string& filename, const std::vector<float>& audio, int sample_rate = 24000) {
    std::ofstream file(filename, std::ios::binary);
    
    // WAV header
    int data_size = audio.size() * 2;
    int file_size = 36 + data_size;
    
    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&file_size), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    
    int fmt_size = 16;
    short format = 1, channels = 1, bits = 16;
    int byte_rate = sample_rate * 2;
    short block_align = 2;
    
    file.write(reinterpret_cast<const char*>(&fmt_size), 4);
    file.write(reinterpret_cast<const char*>(&format), 2);
    file.write(reinterpret_cast<const char*>(&channels), 2);
    file.write(reinterpret_cast<const char*>(&sample_rate), 4);
    file.write(reinterpret_cast<const char*>(&byte_rate), 4);
    file.write(reinterpret_cast<const char*>(&block_align), 2);
    file.write(reinterpret_cast<const char*>(&bits), 2);
    
    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&data_size), 4);
    
    for (float sample : audio) {
        short pcm = static_cast<short>(std::clamp(sample * 32767.0f, -32768.0f, 32767.0f));
        file.write(reinterpret_cast<const char*>(&pcm), 2);
    }
}
```

## Embeddings

### Setup Embeddings

```cpp
bool setupEmbeddings(cactus::cactus_context& context, const std::string& model_path) {
    common_params params;
    params.model.path = model_path;
    params.n_ctx = 512;
    params.n_gpu_layers = 99;
    params.embedding = true;
    params.embd_normalize = 2;
    params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    
    return context.loadModel(params);
}
```

### Generate Embeddings

```cpp
std::vector<float> getEmbedding(cactus::cactus_context& context, const std::string& text) {
    context.rewind();
    context.params.prompt = text;
    context.params.n_predict = 0;
    
    context.initSampling();
    context.beginCompletion();
    context.loadPrompt();
    context.doCompletion();
    
    common_params embd_params;
    embd_params.embd_normalize = context.params.embd_normalize;
    
    return context.getEmbedding(embd_params);
}

float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}
```

### Similarity Search

```cpp
class EmbeddingSearch {
private:
    cactus::cactus_context context;
    std::vector<std::string> texts;
    std::vector<std::vector<float>> embeddings;
    
public:
    bool init(const std::string& model_path) {
        return setupEmbeddings(context, model_path);
    }
    
    void add(const std::string& text) {
        auto embedding = getEmbedding(context, text);
        if (!embedding.empty()) {
            texts.push_back(text);
            embeddings.push_back(embedding);
        }
    }
    
    std::vector<std::pair<std::string, float>> search(const std::string& query, int top_k = 5) {
        auto query_emb = getEmbedding(context, query);
        std::vector<std::pair<std::string, float>> results;
        
        for (size_t i = 0; i < texts.size(); ++i) {
            float sim = cosineSimilarity(query_emb, embeddings[i]);
            results.emplace_back(texts[i], sim);
        }
        
        std::sort(results.begin(), results.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        if (results.size() > static_cast<size_t>(top_k)) {
            results.resize(top_k);
        }
        
        return results;
    }
};
```

## Advanced Usage

### LoRA Adapters

```cpp
bool applyLoRA(cactus::cactus_context& context, const std::vector<std::string>& lora_paths) {
    std::vector<common_adapter_lora_info> adapters;
    
    for (const auto& path : lora_paths) {
        common_adapter_lora_info adapter;
        adapter.path = path;
        adapter.scaled = 1.0f;
        adapters.push_back(adapter);
    }
    
    return context.applyLoraAdapters(adapters) > 0;
}

void removeLoRA(cactus::cactus_context& context) {
    context.removeLoraAdapters();
}
```

### Custom Sampling

```cpp
void configureSampling(common_params& params, float temp, int top_k, float top_p) {
    params.sampling.temp = temp;
    params.sampling.top_k = top_k;
    params.sampling.top_p = top_p;
    params.sampling.penalty_repeat = 1.1f;
    params.sampling.seed = -1; // Random seed
}

void setStopWords(common_params& params, const std::vector<std::string>& stop_words) {
    params.antiprompt = stop_words;
}
```

### Memory Management

```cpp
class ModelManager {
private:
    std::unique_ptr<cactus::cactus_context> context;
    
public:
    ModelManager() : context(std::make_unique<cactus::cactus_context>()) {}
    
    ~ModelManager() {
        if (context) {
            context->releaseMultimodal();
            context->releaseVocoder();
        }
    }
    
    cactus::cactus_context* get() { return context.get(); }
};
```

## API Reference

### Core Methods

**`cactus::cactus_context`**
- `bool loadModel(common_params& params)` - Load model
- `bool initSampling()` - Initialize sampling
- `void rewind()` - Reset state
- `void beginCompletion()` - Start generation
- `void loadPrompt()` - Load text prompt
- `void loadPrompt(const std::vector<std::string>& media_paths)` - Load with media
- `completion_token_output doCompletion()` - Generate next token
- `void endCompletion()` - End generation

### Multimodal Methods

- `bool initMultimodal(const std::string& mmproj_path, bool use_gpu)` - Initialize vision
- `bool isMultimodalEnabled()` - Check status
- `bool isMultimodalSupportVision()` - Check vision support
- `void releaseMultimodal()` - Cleanup

### TTS Methods

- `bool initVocoder(const std::string& vocoder_path)` - Initialize TTS
- `bool isVocoderEnabled()` - Check status
- `std::string getFormattedAudioCompletion(const std::string& speaker, const std::string& text)` - Format prompt
- `std::vector<llama_token> getAudioCompletionGuideTokens(const std::string& text)` - Get guide tokens
- `std::vector<float> decodeAudioTokens(const std::vector<llama_token>& tokens)` - Decode audio
- `void releaseVocoder()` - Cleanup

### Utility Methods

- `std::vector<float> getEmbedding(common_params& params)` - Generate embeddings
- `std::string getFormattedChat(const std::string& messages, const std::string& template)` - Format chat
- `int applyLoraAdapters(std::vector<common_adapter_lora_info> lora)` - Apply LoRA
- `void removeLoraAdapters()` - Remove LoRA
- `void setGuideTokens(const std::vector<llama_token>& tokens)` - Set guide tokens

### Key Parameters

**Model Configuration:**
- `params.model.path` - Model file path
- `params.n_ctx` - Context window size (512-32768)
- `params.n_batch` - Batch size (32-2048)
- `params.n_gpu_layers` - GPU layers (0=CPU, 99=all GPU)
- `params.cpuparams.n_threads` - CPU threads

**Sampling Configuration:**
- `params.sampling.temp` - Temperature (0.1-2.0)
- `params.sampling.top_k` - Top-k sampling (1-100)
- `params.sampling.top_p` - Nucleus sampling (0.1-1.0)
- `params.sampling.penalty_repeat` - Repetition penalty (1.0-1.5)

**Generation Control:**
- `params.n_predict` - Max tokens to generate
- `params.antiprompt` - Stop words
- `params.embedding` - Enable embedding mode
- `params.embd_normalize` - Embedding normalization (0-2)

Check [examples](example/) for complete working code.
