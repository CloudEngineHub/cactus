#import <Foundation/Foundation.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer to cactus_context
typedef void *CactusContextRef;

// Create a new cactus_context instance
CactusContextRef cactus_context_create(void);

// Load a model into the context. Returns 1 on success, 0 on failure.
int cactus_context_load_model(CactusContextRef ctx, const char *model_path);

// Destroy the cactus_context instance
void cactus_context_destroy(CactusContextRef ctx);

#ifdef __cplusplus
}
#endif 