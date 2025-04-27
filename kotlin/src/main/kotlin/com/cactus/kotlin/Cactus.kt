package com.cactus.kotlin

import android.content.Context
import com.cactus.kotlin.listeners.LoadProgressListener
import com.cactus.kotlin.models.ContextParams
import com.cactus.kotlin.models.ModelInfo
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Static entry point for the Cactus library
 */
object Cactus {
    /**
     * Check if the device architecture is supported
     * @return true if the architecture is supported
     */
    fun isArchitectureSupported(): Boolean {
        val arch = System.getProperty("os.arch") ?: return false
        return arch.contains("aarch64") || arch.contains("x86_64")
    }
    
    /**
     * Get information about a model without loading it fully
     * @param modelPath Path to the model file
     * @param skipMetadata List of metadata keys to skip
     * @return ModelInfo object containing model information
     */
    suspend fun getModelInfo(
        modelPath: String,
        skipMetadata: List<String> = emptyList()
    ): ModelInfo {
        return LlamaContext.getModelInfo(modelPath, skipMetadata)
    }
    
    /**
     * Create a new LlamaContext for LLM operations
     * @param params The context parameters
     * @param androidContext Optional Android context for file access
     * @param progressListener Optional listener for load progress
     * @return A new LlamaContext instance
     */
    suspend fun createContext(
        params: ContextParams,
        androidContext: Context? = null,
        progressListener: LoadProgressListener? = null
    ): LlamaContext {
        return LlamaContext.create(params, androidContext, progressListener)
    }
    
    /**
     * Toggle native logging
     * To avoid memory leaks, turn this off when you're done debugging
     * @param enabled Whether to enable native logging
     */
    suspend fun toggleNativeLogging(enabled: Boolean): Unit = withContext(Dispatchers.IO) {
        // This would call a JNI function to toggle logging
        // Implementation depends on your native logging setup
    }
    
    /**
     * Check if GGML uses a specific CPU feature
     * @param feature The CPU feature to check
     * @return true if the feature is used
     */
    fun hasCpuFeature(feature: CpuFeature): Boolean {
        val arch = System.getProperty("os.arch") ?: return false
        
        return when {
            arch.contains("aarch64") -> {
                when (feature) {
                    CpuFeature.AVX -> false
                    CpuFeature.AVX2 -> false
                    CpuFeature.AVX512 -> false
                    CpuFeature.NEON -> true
                    CpuFeature.DOT_PROD -> false // Would need to check at runtime
                    CpuFeature.INT8_MM -> false  // Would need to check at runtime
                }
            }
            arch.contains("x86_64") -> {
                when (feature) {
                    CpuFeature.AVX -> true
                    CpuFeature.AVX2 -> true
                    CpuFeature.AVX512 -> false  // Would need to check at runtime
                    CpuFeature.NEON -> false
                    CpuFeature.DOT_PROD -> false
                    CpuFeature.INT8_MM -> false
                }
            }
            else -> false
        }
    }
}

/**
 * CPU features that can be used by GGML
 */
enum class CpuFeature {
    AVX,
    AVX2,
    AVX512,
    NEON,
    DOT_PROD,
    INT8_MM
} 