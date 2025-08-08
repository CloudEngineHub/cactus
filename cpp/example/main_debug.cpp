#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <iomanip>

#include "utils.h"
#include "../../cpp/cactus.h"
#include "../../cpp/llama.h"
#include "../../cpp/ggml.h"

struct callback_data {
    std::vector<uint8_t> data;
    int target_layer = 1;  // We want layer 1 output
    bool found_layer_1 = false;
};

static std::string ggml_ne_string(const lm_ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < LM_GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < LM_GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static void print_tensor_summary(uint8_t * data, lm_ggml_type type, const int64_t * ne, const size_t * nb, const std::string& name) {
    if (!data) return;
    
    std::cout << name << " [" << ggml_ne_string((lm_ggml_tensor*)nullptr) << "]:" << std::endl;
    
    // Calculate total elements for first dimension
    int64_t n_elements = std::min((int64_t)10, ne[0]);  // Show first 10 values
    
    std::cout << "  First " << n_elements << " values: [";
    for (int64_t i = 0; i < n_elements; i++) {
        float v = 0;
        size_t offset = i * nb[0];
        
        if (type == LM_GGML_TYPE_F16) {
            v = lm_ggml_fp16_to_fp32(*(lm_ggml_fp16_t *) &data[offset]);
        } else if (type == LM_GGML_TYPE_F32) {
            v = *(float *) &data[offset];
        }
        
        std::cout << std::fixed << std::setprecision(4) << v;
        if (i < n_elements - 1) std::cout << ", ";
    }
    std::cout << ", ...]" << std::endl;
    
    // Calculate basic statistics for first 100 elements
    int64_t stats_elements = std::min((int64_t)100, ne[0]);
    float sum = 0, min_val = 0, max_val = 0;
    bool first = true;
    
    for (int64_t i = 0; i < stats_elements; i++) {
        float v = 0;
        size_t offset = i * nb[0];
        
        if (type == LM_GGML_TYPE_F16) {
            v = lm_ggml_fp16_to_fp32(*(lm_ggml_fp16_t *) &data[offset]);
        } else if (type == LM_GGML_TYPE_F32) {
            v = *(float *) &data[offset];
        }
        
        sum += v;
        if (first || v < min_val) min_val = v;
        if (first || v > max_val) max_val = v;
        first = false;
    }
    
    float mean = stats_elements > 0 ? sum / stats_elements : 0;
    std::cout << "  Stats (first " << stats_elements << "): min=" << min_val 
              << ", max=" << max_val << ", mean=" << mean << std::endl << std::endl;
}

static bool ggml_debug_callback(struct lm_ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    if (ask) {
        // We're interested in layer 1 tensors and key intermediate outputs
        std::string name = t->name ? t->name : "unnamed";
        
        // Debug: Let's be more permissive initially to see what tensors exist
        // Look for layer 1 tensors with various possible naming schemes
        if (name.find("blk.1.") != std::string::npos ||  // blk.1.xxx
            name.find("layer.1") != std::string::npos ||  // layer.1.xxx  
            name.find("layers.1") != std::string::npos || // layers.1.xxx
            name.find(".1.") != std::string::npos ||      // xxx.1.xxx
            (name.find("1") != std::string::npos && name.find("attn") != std::string::npos) ||
            (name.find("1") != std::string::npos && name.find("ffn") != std::string::npos)) {
            return true;  // We want this tensor
        }
        
        // Also capture embedding and important base tensors
        if (name.find("token") != std::string::npos ||
            name.find("embd") != std::string::npos ||
            name.find("inp") != std::string::npos ||
            name.find("embeddings") != std::string::npos) {
            return true;
        }
        
        // Capture first 10 tensors for debugging naming patterns
        static int tensor_count = 0;
        if (tensor_count < 10) {
            tensor_count++;
            return true;  // Capture first few to see naming scheme
        }
        
        return false;  // Skip other tensors
    }

    // We're getting the actual data
    std::string name = t->name ? t->name : "unnamed";
    
    std::cout << "=== Captured Tensor: " << name << " ===" << std::endl;
    std::cout << "Type: " << lm_ggml_type_name(t->type) << std::endl;
    std::cout << "Shape: [" << ggml_ne_string(t) << "]" << std::endl;

    // Copy tensor data if it's on GPU
    const bool is_host = lm_ggml_backend_buffer_is_host(t->buffer);
    
    if (!is_host) {
        auto n_bytes = lm_ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        lm_ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    // Print tensor summary for non-quantized tensors
    if (!lm_ggml_is_quantized(t->type)) {
        uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
        print_tensor_summary(data, t->type, t->ne, t->nb, name);
    }
    
    // Mark that we found layer 1 output
    if (name.find("blk.1.") != std::string::npos) {
        cb_data->found_layer_1 = true;
    }

    return true;  // Continue processing
}

void print_tokens(const std::vector<llama_token>& tokens, llama_model* model) {
    std::cout << "\n=== Tokenizer Output ===" << std::endl;
    std::cout << "Token IDs: [";
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cout << tokens[i];
        if (i < tokens.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Shape: [" << tokens.size() << "]" << std::endl;
    
    // Print token strings
    std::cout << "Token strings: [";
    const llama_vocab* vocab = llama_model_get_vocab(model);
    for (size_t i = 0; i < tokens.size(); i++) {
        char buf[128];
        int n = llama_token_to_piece(vocab, tokens[i], buf, sizeof(buf), 0, false);
        if (n > 0) {
            std::string token_str(buf, n);
            std::cout << "\"" << token_str << "\"";
        } else {
            std::cout << "<unk>";
        }
        if (i < tokens.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

int main(int argc, char **argv) {
    const std::string model_url = "https://huggingface.co/Cactus-Compute/Qwen3-600m-Instruct-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf?download=true";
    const std::string model_filename = "Qwen3-0.6B-Q8_0.gguf";
    
    if (!downloadFile(model_url, model_filename, "LLM model")) {
        return 1;
    }
    
    std::cout << "\n=== Debug: Layer-wise Analysis ===" << std::endl;
    
    try {
        callback_data cb_data;
        
        cactus::cactus_context context;
        
        common_params params;
        params.model.path = model_filename;
        params.n_ctx = 4096;
        params.n_batch = 512;
        params.n_gpu_layers = 0;
        
        // Set the debug callback - this is the key part!
        params.cb_eval = ggml_debug_callback;
        params.cb_eval_user_data = &cb_data;
        params.warmup = false;  // Disable warmup to see actual computation
        
        std::cout << "Loading model: " << model_filename << std::endl;
        if (!context.loadModel(params)) {
            std::cerr << "Failed to load model" << std::endl;
            return 1;
        }
        
        std::cout << "Model loaded successfully!" << std::endl;
        
        // Get model info
        const llama_vocab* vocab = llama_model_get_vocab(context.model);
        int n_embd = llama_n_embd(context.model);
        int n_layer = llama_n_layer(context.model);
        int n_vocab = llama_n_vocab(vocab);
        
        std::cout << "\n=== Model Info ===" << std::endl;
        std::cout << "Embedding dimension: " << n_embd << std::endl;
        std::cout << "Number of layers: " << n_layer << std::endl;
        std::cout << "Vocabulary size: " << n_vocab << std::endl;
        
        // Tokenize the prompt
        const std::string prompt = "My name is Henry Ndubuaku";
        context.params.prompt = prompt;
        context.params.n_predict = 0;  // No token generation
        
        if (!context.initSampling()) {
            std::cerr << "Failed to initialize sampling" << std::endl;
            return 1;
        }
        
        context.beginCompletion();
        
        std::cout << "\n=== Running Forward Pass with Debug Hooks ===" << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // This will trigger our debug callback for each tensor operation
        context.loadPrompt();  
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto prefill_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // Now print tokenizer output (after loadPrompt processes the tokens)
        print_tokens(context.embd, context.model);
        
        std::cout << "\n=== Performance ===" << std::endl;
        std::cout << "Tokens processed: " << context.num_prompt_tokens << std::endl;
        std::cout << "Prefill time: " << prefill_time.count() / 1000.0 << " ms" << std::endl;
        if (context.num_prompt_tokens > 0 && prefill_time.count() > 0) {
            float tokens_per_second = (float)context.num_prompt_tokens * 1000000.0f / prefill_time.count();
            std::cout << "Prefill speed: " << tokens_per_second << " tok/s" << std::endl;
        }
        
        if (cb_data.found_layer_1) {
            std::cout << "\n✅ Successfully captured layer 1 outputs!" << std::endl;
        } else {
            std::cout << "\n⚠️  No layer 1 outputs detected - check tensor naming" << std::endl;
        }
        
        context.endCompletion();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}