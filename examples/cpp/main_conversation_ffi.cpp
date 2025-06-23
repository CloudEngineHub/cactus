#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <chrono>

#include "utils.h"
#include "../../cactus/cactus_ffi.h"

void print_performance_metrics(const cactus_conversation_result_c_t& result) {
    std::cout << "[PERFORMANCE] TTFT: " << result.time_to_first_token << "ms, "
              << "Total: " << result.total_time << "ms, "
              << "Tokens: " << result.tokens_generated;
    
    if (result.tokens_generated > 0 && result.total_time > 0) {
        float tokens_per_second = (float)result.tokens_generated * 1000.0f / result.total_time;
        std::cout << ", Speed: " << std::fixed << std::setprecision(1) 
                  << tokens_per_second << " tok/s";
    }
    std::cout << std::endl;
}

bool conversation_demo(cactus_context_handle_t handle) {
    std::cout << "\n=== Conversation Management Demo ===" << std::endl;
    
    std::vector<std::string> messages = {
        "Hello! How are you today?",
        "What can you help me with?", 
        "Tell me a fun fact about space",
        "Can you explain that in simpler terms?",
        "Thank you for the explanation!"
    };
    
    for (size_t i = 0; i < messages.size(); ++i) {
        std::cout << "\nTurn " << (i + 1) << ":" << std::endl;
        std::cout << "User: " << messages[i] << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        cactus_conversation_result_c_t result = cactus_continue_conversation_c(handle, messages[i].c_str(), 150);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto js_overhead = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (!result.text) {
            std::cerr << "Failed to get response for message: " << messages[i] << std::endl;
            return false;
        }
        
        std::cout << "Assistant: " << result.text << std::endl;
        print_performance_metrics(result);
        std::cout << "[TIMING] C++ FFI overhead: " << (js_overhead.count() - result.total_time) << "ms" << std::endl;
        
        // Check conversation status
        bool is_active = cactus_is_conversation_active_c(handle);
        std::cout << "[STATUS] Conversation active: " << (is_active ? "Yes" : "No") << std::endl;
        
        cactus_free_conversation_result_members_c(&result);
        
        std::cout << std::string(60, '-') << std::endl;
    }
    
    return true;
}

bool simple_response_demo(cactus_context_handle_t handle) {
    std::cout << "\n=== Simple Response Demo ===" << std::endl;
    
    std::vector<std::string> prompts = {
        "Write a haiku about programming",
        "What is the meaning of life?",
        "Explain quantum computing in one sentence"
    };
    
    for (const auto& prompt : prompts) {
        std::cout << "\nPrompt: " << prompt << std::endl;
        
        char* response = cactus_generate_response_c(handle, prompt.c_str(), 100);
        
        if (!response) {
            std::cerr << "Failed to generate response" << std::endl;
            return false;
        }
        
        std::cout << "Response: " << response << std::endl;
        cactus_free_string_c(response);
        
        std::cout << std::string(50, '-') << std::endl;
    }
    
    return true;
}

int main(int argc, char **argv) {
    const std::string model_url = "https://huggingface.co/QuantFactory/SmolLM-360M-Instruct-GGUF/resolve/main/SmolLM-360M-Instruct.Q6_K.gguf";
    const std::string model_filename = "SmolLM-360M-Instruct.Q6_K.gguf";
    
    if (!downloadFile(model_url, model_filename, "LLM model")) {
        return 1;
    }
    
    std::cout << "\n=== Cactus Conversation FFI Example ===" << std::endl;
    
    try {
        // Initialize context using FFI
        cactus_init_params_c_t init_params = {};
        init_params.model_path = model_filename.c_str();
        init_params.chat_template = nullptr;
        init_params.n_ctx = 2048;
        init_params.n_batch = 512;
        init_params.n_ubatch = 512;
        init_params.n_gpu_layers = 99; // Use GPU acceleration
        init_params.n_threads = 4;
        init_params.use_mmap = true;
        init_params.use_mlock = false;
        init_params.embedding = false;
        init_params.pooling_type = 0;
        init_params.embd_normalize = 2;
        init_params.flash_attn = false;
        init_params.cache_type_k = nullptr;
        init_params.cache_type_v = nullptr;
        init_params.progress_callback = nullptr;

        std::cout << "Loading model: " << model_filename << std::endl;
        cactus_context_handle_t handle = cactus_init_context_c(&init_params);
        if (!handle) {
            std::cerr << "Failed to load model" << std::endl;
            return 1;
        }

        std::cout << "Model loaded successfully!" << std::endl;
        
        // Get model information
        char* model_desc = cactus_get_model_desc_c(handle);
        int32_t n_ctx = cactus_get_n_ctx_c(handle);
        std::cout << "Model: " << (model_desc ? model_desc : "Unknown") << std::endl;
        std::cout << "Context size: " << n_ctx << std::endl;
        cactus_free_string_c(model_desc);
        
        // Run different demos based on command line arguments
        if (argc > 1 && std::string(argv[1]) == "simple") {
            if (!simple_response_demo(handle)) {
                cactus_free_context_c(handle);
                return 1;
            }
        } else if (argc > 1 && std::string(argv[1]) == "conversation") {
            if (!conversation_demo(handle)) {
                cactus_free_context_c(handle);
                return 1;
            }
        } else {
            std::cout << "\nAvailable demos:" << std::endl;
            std::cout << "  ./conversation_ffi simple       - Simple generateResponse demo" << std::endl;
            std::cout << "  ./conversation_ffi conversation - Full conversation management demo" << std::endl;
            std::cout << "\nNew Conversation API Features:" << std::endl;
            std::cout << "  - Automatic KV cache optimization" << std::endl;
            std::cout << "  - Consistent TTFT across conversation turns" << std::endl;
            std::cout << "  - Built-in performance timing" << std::endl;
            std::cout << "  - Simple conversation state management" << std::endl;
            std::cout << "\nRunning conversation demo by default...\n" << std::endl;
            
            if (!conversation_demo(handle)) {
                cactus_free_context_c(handle);
                return 1;
            }
        }
        
        // Clear conversation and test state
        std::cout << "\nClearing conversation..." << std::endl;
        cactus_clear_conversation_c(handle);
        bool is_active = cactus_is_conversation_active_c(handle);
        std::cout << "Conversation active after clear: " << (is_active ? "Yes" : "No") << std::endl;

        // Clean up
        cactus_free_context_c(handle);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 