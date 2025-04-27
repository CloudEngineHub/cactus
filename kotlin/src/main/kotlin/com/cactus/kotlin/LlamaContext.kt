package com.cactus.kotlin

import android.content.Context
import com.cactus.kotlin.listeners.CompletionListener
import com.cactus.kotlin.listeners.LoadProgressListener
import com.cactus.kotlin.models.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Main class for interacting with the Cactus LLM engine
 */
class LlamaContext private constructor(
    private val contextId: Long,
    private val modelInfo: ModelInfo
) {
    private val isReleased = AtomicBoolean(false)
    
    companion object {
        init {
            try {
                System.loadLibrary("cactus")
            } catch (e: UnsatisfiedLinkError) {
                // Try other optimized libraries based on architecture
                try {
                    if (isArm64V8a()) {
                        // Try different ARM optimizations in order of capability
                        val libraries = listOf(
                            "cactus_v8_2_dotprod_i8mm",
                            "cactus_v8_2_dotprod",
                            "cactus_v8_2_i8mm",
                            "cactus_v8_2",
                            "cactus_v8"
                        )
                        var loaded = false
                        for (lib in libraries) {
                            try {
                                System.loadLibrary(lib)
                                loaded = true
                                break
                            } catch (e: UnsatisfiedLinkError) {
                                // Continue trying
                            }
                        }
                        if (!loaded) {
                            throw UnsatisfiedLinkError("Could not load any cactus library for ARM64")
                        }
                    } else if (isX86_64()) {
                        System.loadLibrary("cactus_x86_64")
                    } else {
                        throw UnsatisfiedLinkError("Architecture not supported")
                    }
                } catch (e: UnsatisfiedLinkError) {
                    throw UnsatisfiedLinkError("Failed to load cactus native library: ${e.message}")
                }
            }
        }
        
        /**
         * Create a new LlamaContext from parameters
         */
        suspend fun create(
            params: ContextParams,
            androidContext: Context? = null,
            progressListener: LoadProgressListener? = null
        ): LlamaContext = withContext(Dispatchers.IO) {
            val jniLoadProgressCallback = progressListener?.let { listener ->
                object : JNIBridge.LoadProgressCallback {
                    override fun onProgress(progress: Int) {
                        listener.onProgress(progress)
                    }
                }
            }
            
            val contextPtr = JNIBridge.initContext(
                params.model,
                params.chatTemplate ?: "",
                params.reasoningFormat,
                params.embedding,
                params.embdNormalize,
                params.nCtx,
                params.nBatch,
                params.nUbatch,
                params.nThreads,
                params.nGpuLayers,
                params.flashAttn,
                params.cacheTypeK,
                params.cacheTypeV,
                params.useMlock,
                params.useMmap,
                params.vocabOnly,
                params.lora ?: "",
                params.loraScaled,
                params.loraList?.toTypedArray() ?: emptyArray(),
                params.ropeFreqBase,
                params.ropeFreqScale,
                params.poolingType,
                jniLoadProgressCallback
            )
            
            if (contextPtr <= 0) {
                throw IllegalStateException("Failed to initialize LlamaContext")
            }
            
            val modelInfoMap = JNIBridge.loadModelDetails(contextPtr)
            val modelInfo = ModelInfo(
                name = modelInfoMap["name"] ?: "Unknown",
                architecture = modelInfoMap["arch"] ?: "Unknown",
                params = modelInfoMap["params"]?.toLongOrNull() ?: 0L,
                contextSize = modelInfoMap["contextSize"]?.toIntOrNull() ?: 0,
                vocabSize = modelInfoMap["vocabSize"]?.toIntOrNull() ?: 0,
                embeddingSize = modelInfoMap["embeddingSize"]?.toIntOrNull() ?: 0,
                supportsEmbeddings = modelInfoMap["embedding"]?.toBoolean() ?: false,
                metadata = modelInfoMap.filter { (key, _) -> 
                    !listOf("name", "arch", "params", "contextSize", "vocabSize", "embeddingSize", "embedding").contains(key)
                }
            )
            
            progressListener?.onComplete()
            
            LlamaContext(contextPtr, modelInfo)
        }
        
        /**
         * Get information about a model without loading it fully
         */
        suspend fun getModelInfo(
            modelPath: String,
            skipMetadata: List<String> = emptyList()
        ): ModelInfo = withContext(Dispatchers.IO) {
            val modelInfoMap = JNIBridge.modelInfo(modelPath, skipMetadata.toTypedArray())
            
            ModelInfo(
                name = modelInfoMap["name"] ?: "Unknown",
                architecture = modelInfoMap["arch"] ?: "Unknown",
                params = modelInfoMap["params"]?.toLongOrNull() ?: 0L,
                contextSize = modelInfoMap["contextSize"]?.toIntOrNull() ?: 0,
                vocabSize = modelInfoMap["vocabSize"]?.toIntOrNull() ?: 0,
                embeddingSize = modelInfoMap["embeddingSize"]?.toIntOrNull() ?: 0,
                supportsEmbeddings = modelInfoMap["embedding"]?.toBoolean() ?: false,
                metadata = modelInfoMap.filter { (key, _) -> 
                    !listOf("name", "arch", "params", "contextSize", "vocabSize", "embeddingSize", "embedding").contains(key)
                }
            )
        }
        
        private fun isArm64V8a(): Boolean {
            return System.getProperty("os.arch")?.contains("aarch64") == true
        }
        
        private fun isX86_64(): Boolean {
            return System.getProperty("os.arch")?.contains("x86_64") == true
        }
    }
    
    /**
     * Get model information
     */
    fun getModelInfo(): ModelInfo = modelInfo
    
    /**
     * Generate completion from a prompt
     */
    suspend fun completion(
        params: CompletionParams,
        listener: CompletionListener? = null
    ): CompletionResult = withContext(Dispatchers.IO) {
        checkReleased()
        
        val jniCompletionCallback = listener?.let { listen ->
            object : JNIBridge.TokenCallback {
                override fun onToken(tokenText: String, isPartial: Boolean, tokenId: Int) {
                    listen.onToken(tokenText, isPartial, tokenId)
                }
            }
        }
        
        val result = JNIBridge.doCompletion(
            contextId,
            params.prompt,
            params.maxTokens,
            params.temperature,
            params.topP,
            params.topK,
            params.frequencyPenalty,
            params.presencePenalty,
            params.repetitionPenalty,
            params.mirostat,
            params.mirostatTau,
            params.mirostatEta,
            params.grammar ?: "",
            params.jsonSchema ?: "",
            params.stopSequences?.toTypedArray() ?: emptyArray(),
            params.seed,
            params.logitBias ?: emptyMap(),
            params.system ?: "",
            params.template ?: "",
            params.trtEnabled,
            jniCompletionCallback
        )
        
        listener?.onComplete()
        
        CompletionResult(
            text = result["text"] ?: "",
            timeTaken = result["timeTaken"]?.toLongOrNull() ?: 0L,
            totalTokens = result["totalTokens"]?.toIntOrNull() ?: 0,
            tokensPerSecond = result["tokensPerSecond"]?.toFloatOrNull() ?: 0f,
            promptTokens = result["promptTokens"]?.toIntOrNull() ?: 0,
            completionTokens = result["completionTokens"]?.toIntOrNull() ?: 0,
            truncated = result["truncated"]?.toBoolean() ?: false,
            stopReason = result["stopReason"]
        )
    }
    
    /**
     * Stop an ongoing completion
     */
    suspend fun stopCompletion(): Unit = withContext(Dispatchers.IO) {
        checkReleased()
        JNIBridge.stopCompletion(contextId)
    }
    
    /**
     * Tokenize text into token IDs
     */
    suspend fun tokenize(text: String): List<Int> = withContext(Dispatchers.IO) {
        checkReleased()
        JNIBridge.tokenize(contextId, text).toList()
    }
    
    /**
     * Convert token IDs back to text
     */
    suspend fun detokenize(tokens: List<Int>): String = withContext(Dispatchers.IO) {
        checkReleased()
        JNIBridge.detokenize(contextId, tokens.toIntArray())
    }
    
    /**
     * Get text embedding
     */
    suspend fun getEmbedding(text: String): FloatArray = withContext(Dispatchers.IO) {
        checkReleased()
        
        if (!JNIBridge.isEmbeddingEnabled(contextId)) {
            throw IllegalStateException("Embeddings not enabled for this model")
        }
        
        val result = JNIBridge.embedding(contextId, text, emptyMap<String, Any>())
        result["embedding"]?.let { embeddingData ->
            (embeddingData as? List<*>)?.mapNotNull { (it as? Number)?.toFloat() }?.toFloatArray()
                ?: throw IllegalStateException("Invalid embedding format")
        } ?: throw IllegalStateException("Failed to get embedding")
    }
    
    /**
     * Format chat messages with the model's template
     */
    suspend fun formatChat(
        messages: List<Map<String, String>>,
        customTemplate: String? = null
    ): String = withContext(Dispatchers.IO) {
        checkReleased()
        
        val messagesJson = messages.joinToString(",") { message ->
            val role = message["role"] ?: throw IllegalArgumentException("Message must have a role")
            val content = message["content"] ?: throw IllegalArgumentException("Message must have content")
            """{"role":"$role","content":"$content"}"""
        }
        val messagesArray = "[$messagesJson]"
        
        JNIBridge.getFormattedChat(contextId, messagesArray, customTemplate ?: "")
    }
    
    /**
     * Apply LoRA adapters to the model
     */
    suspend fun applyLoraAdapters(adapters: List<String>): Int = withContext(Dispatchers.IO) {
        checkReleased()
        JNIBridge.applyLoraAdapters(contextId, adapters.toTypedArray())
    }
    
    /**
     * Remove all applied LoRA adapters
     */
    suspend fun removeLoraAdapters(): Unit = withContext(Dispatchers.IO) {
        checkReleased()
        JNIBridge.removeLoraAdapters(contextId)
    }
    
    /**
     * Get list of currently loaded LoRA adapters
     */
    suspend fun getLoadedLoraAdapters(): List<String> = withContext(Dispatchers.IO) {
        checkReleased()
        JNIBridge.getLoadedLoraAdapters(contextId).toList()
    }
    
    /**
     * Save the current context state to a file
     */
    suspend fun saveSession(path: String, size: Int = 0): Int = withContext(Dispatchers.IO) {
        checkReleased()
        JNIBridge.saveSession(contextId, path, size)
    }
    
    /**
     * Load context state from a file
     */
    suspend fun loadSession(path: String): Map<String, String> = withContext(Dispatchers.IO) {
        checkReleased()
        JNIBridge.loadSession(contextId, path)
    }
    
    /**
     * Release native resources
     */
    suspend fun release(): Unit = withContext(Dispatchers.IO) {
        if (isReleased.compareAndSet(false, true)) {
            JNIBridge.freeContext(contextId)
        }
    }
    
    private fun checkReleased() {
        if (isReleased.get()) {
            throw IllegalStateException("Context has been released")
        }
    }
} 