package com.cactus.kotlin;

import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

/**
 * JNI Bridge for the Cactus library.
 * This class provides native method implementations that are called from Kotlin.
 * It serves as an adapter between the Kotlin interface and the existing JNI code.
 */
class JNIBridge {
    static {
        try {
            System.loadLibrary("cactus");
        } catch (UnsatisfiedLinkError e) {
            // Native library loading is handled in LlamaContext to try various optimized versions
        }
    }
    
    // Model initialization and information
    public static native Map<String, String> modelInfo(String modelPath, String[] skipMetadata);
    
    public static native long initContext(
        String model,
        String chatTemplate,
        String reasoningFormat,
        boolean embedding,
        int embdNormalize,
        int nCtx,
        int nBatch,
        int nUbatch,
        int nThreads,
        int nGpuLayers,
        boolean flashAttn,
        String cacheTypeK,
        String cacheTypeV,
        boolean useMlock,
        boolean useMmap,
        boolean vocabOnly,
        String lora,
        float loraScaled,
        String[] loraList,
        float ropeFreqBase,
        float ropeFreqScale,
        int poolingType,
        LoadProgressCallback loadProgressCallback
    );
    
    public static native Map<String, String> loadModelDetails(long contextPtr);
    
    // Completion
    public static native Map<String, String> doCompletion(
        long contextPtr,
        String prompt,
        int maxTokens,
        float temperature,
        float topP,
        int topK,
        float frequencyPenalty,
        float presencePenalty,
        float repetitionPenalty,
        int mirostat,
        float mirostatTau,
        float mirostatEta,
        String grammar,
        String jsonSchema,
        String[] stopSequences,
        int seed,
        Map<Integer, Float> logitBias,
        String system,
        String template,
        boolean trtEnabled,
        TokenCallback tokenCallback
    );
    
    public static native void stopCompletion(long contextPtr);
    
    public static native boolean isPredicting(long contextPtr);
    
    // Token handling
    public static native int[] tokenize(long contextPtr, String text);
    
    public static native String detokenize(long contextPtr, int[] tokens);
    
    // Embeddings
    public static native boolean isEmbeddingEnabled(long contextPtr);
    
    public static native Map<String, Object> embedding(
        long contextPtr,
        String text,
        Map<String, Object> params
    );
    
    // Chat formatting
    public static native String getFormattedChat(
        long contextPtr,
        String messages,
        String chatTemplate
    );
    
    public static native Map<String, Object> getFormattedChatWithJinja(
        long contextPtr,
        String messages,
        String chatTemplate,
        String jsonSchema,
        String tools,
        boolean parallelToolCalls,
        String toolChoice
    );
    
    // LoRA adapters
    public static native int applyLoraAdapters(long contextPtr, String[] loraAdapters);
    
    public static native void removeLoraAdapters(long contextPtr);
    
    public static native String[] getLoadedLoraAdapters(long contextPtr);
    
    // Session management
    public static native int saveSession(long contextPtr, String path, int size);
    
    public static native Map<String, String> loadSession(long contextPtr, String path);
    
    // Resource management
    public static native void freeContext(long contextPtr);
    
    public static native void setupLog(LogCallback logCallback);
    
    public static native void unsetLog();
    
    // Callback interfaces
    public interface LoadProgressCallback {
        void onProgress(int progress);
    }
    
    public interface TokenCallback {
        void onToken(String tokenText, boolean isPartial, int tokenId);
    }
    
    public interface LogCallback {
        void onLog(String level, String text);
    }
} 