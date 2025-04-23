#include <iostream>
#include <string>
#include <vector>
#include "../cactus/cactus.h"

int main() {
    // Short-prompt response test
    std::string prompt = "Hello, how are you?";

    common_params params;
    params.model      = "../llm.gguf";  // Model is in the same directory as the test executable
    params.prompt     = prompt;           // Set the prompt
    params.n_predict  = 50;               // Number of tokens to generate
    params.n_ctx      = 1024;
    params.n_batch    = 512;
    params.cpuparams.n_threads = 4;
    params.use_mmap   = true;
    params.warmup     = false;

    // Initialize llama context
    cactus::cactus_context ctx;
    if (!ctx.loadModel(params)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    std::cout << "Model loaded successfully" << std::endl;

    // Initialize sampling
    if (!ctx.initSampling()) {
        std::cerr << "Failed to initialize sampling" << std::endl;
        return 1;
    }

    // Encode prompt and start completion
    ctx.loadPrompt();
    ctx.beginCompletion();

    std::cout << "\nPrompt: " << prompt << std::endl;
    std::cout << "Response: ";

    const llama_vocab * vocab = llama_model_get_vocab(ctx.model);
    while (ctx.has_next_token) {
        auto tok = ctx.nextToken();
        if (tok.tok < 0) break;
        if (tok.tok == llama_vocab_eos(vocab)) break;

        char buffer[64];
        int length = llama_token_to_piece(vocab, tok.tok, buffer, sizeof(buffer), false, false);
        if (length > 0) {
            std::cout << std::string(buffer, length) << std::flush;
        }
    }
    std::cout << std::endl;

    return 0;
} 