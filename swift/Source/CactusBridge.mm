#import "CactusBridge.h"

// Import C++ headers from the xcframework
#include <cactus.h>
#include <common.h>

using namespace cactus;

CactusContextRef cactus_context_create(void) {
    return new cactus_context();
}

int cactus_context_load_model(CactusContextRef ctxRef, const char *model_path) {
    if (!ctxRef || !model_path) return 0;
    cactus_context *ctx = static_cast<cactus_context *>(ctxRef);
    common_params params;
    params.model = std::string(model_path);
    return ctx->loadModel(params) ? 1 : 0;
}

void cactus_context_destroy(CactusContextRef ctxRef) {
    if (ctxRef) {
        cactus_context *ctx = static_cast<cactus_context *>(ctxRef);
        delete ctx;
    }
} 