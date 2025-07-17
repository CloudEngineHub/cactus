package com.cactus

class Cactus {
    private var handle: CactusContextHandle? = null
    
    suspend fun initialize(params: CactusInitParams): Boolean {
        return try {
            val contextHandle = CactusContext.initContext(params) ?: 0L
            if (contextHandle != 0L) {
                handle = contextHandle
                true
            } else {
                false
            }
        } catch (e: Exception) {
            false
        }
    }
    
    fun close() {
        handle?.let { h ->
            CactusContext.freeContext(h)
            handle = null
        }
    }
    
    suspend fun completion(params: CactusCompletionParams): CactusCompletionResult? {
        return handle?.let { h ->
            try {
                CactusContext.completion(h, params)
            } catch (e: Exception) {
                null
            }
        }
    }
    
    suspend fun multimodalCompletion(params: CactusCompletionParams, mediaPaths: List<String>): CactusCompletionResult? {
        return handle?.let { h ->
            try {
                CactusContext.multimodalCompletion(h, params, mediaPaths)
            } catch (e: Exception) {
                null
            }
        }
    }
    
    fun stopCompletion() {
        handle?.let { h ->
            CactusContext.stopCompletion(h)
        }
    }
    
    suspend fun tokenize(text: String): CactusTokenArray? {
        return handle?.let { h ->
            try {
                CactusContext.tokenize(h, text)
            } catch (e: Exception) {
                null
            }
        }
    }
    
    suspend fun detokenize(tokens: IntArray): String? {
        return handle?.let { h ->
            try {
                CactusContext.detokenize(h, tokens)
            } catch (e: Exception) {
                null
            }
        }
    }
    
    suspend fun embedding(text: String): CactusFloatArray? {
        return handle?.let { h ->
            try {
                CactusContext.embedding(h, text)
            } catch (e: Exception) {
                null
            }
        }
    }
    
    suspend fun initMultimodal(mmprojPath: String, useGpu: Boolean = true): Boolean {
        return handle?.let { h ->
            try {
                CactusContext.initMultimodal(h, mmprojPath, useGpu) == 0
            } catch (e: Exception) {
                false
            }
        } ?: false
    }
    
    fun isMultimodalEnabled(): Boolean {
        return handle?.let { h ->
            try {
                CactusContext.isMultimodalEnabled(h)
            } catch (e: Exception) {
                false
            }
        } ?: false
    }
    
    fun supportsVision(): Boolean {
        return handle?.let { h ->
            try {
                CactusContext.supportsVision(h)
            } catch (e: Exception) {
                false
            }
        } ?: false
    }
    
    fun supportsAudio(): Boolean {
        return handle?.let { h ->
            try {
                CactusContext.supportsAudio(h)
            } catch (e: Exception) {
                false
            }
        } ?: false
    }
    
    suspend fun initVocoder(vocoderModelPath: String): Boolean {
        return handle?.let { h ->
            try {
                CactusContext.initVocoder(h, vocoderModelPath) == 0
            } catch (e: Exception) {
                false
            }
        } ?: false
    }
    
    fun isVocoderEnabled(): Boolean {
        return handle?.let { h ->
            try {
                CactusContext.isVocoderEnabled(h)
            } catch (e: Exception) {
                false
            }
        } ?: false
    }
    
    suspend fun bench(pp: Int = 512, tg: Int = 128, pl: Int = 1, nr: Int = 1): CactusBenchResult? {
        return handle?.let { h ->
            try {
                CactusContext.bench(h, pp, tg, pl, nr)
            } catch (e: Exception) {
                null
            }
        }
    }
    
    suspend fun applyLoraAdapters(adapters: List<CactusLoraAdapter>): Boolean {
        return handle?.let { h ->
            try {
                CactusContext.applyLoraAdapters(h, adapters) == 0
            } catch (e: Exception) {
                false
            }
        } ?: false
    }
    
    fun removeLoraAdapters() {
        handle?.let { h ->
            CactusContext.removeLoraAdapters(h)
        }
    }
    
    fun getLoadedLoraAdapters(): List<CactusLoraAdapter> {
        return handle?.let { h ->
            try {
                CactusContext.getLoadedLoraAdapters(h)
            } catch (e: Exception) {
                emptyList()
            }
        } ?: emptyList()
    }
    
    fun validateChatTemplate(useJinja: Boolean = false, name: String? = null): Boolean {
        return handle?.let { h ->
            try {
                CactusContext.validateChatTemplate(h, useJinja, name)
            } catch (e: Exception) {
                false
            }
        } ?: false
    }
    
    suspend fun getFormattedChat(messages: String, chatTemplate: String? = null): String? {
        return handle?.let { h ->
            try {
                CactusContext.getFormattedChat(h, messages, chatTemplate)
            } catch (e: Exception) {
                null
            }
        }
    }
    
    suspend fun getFormattedChatWithJinja(
        messages: String, 
        chatTemplate: String? = null, 
        jsonSchema: String? = null, 
        tools: String? = null, 
        parallelToolCalls: Boolean = false, 
        toolChoice: String? = null
    ): CactusChatResult? {
        return handle?.let { h ->
            try {
                CactusContext.getFormattedChatWithJinja(h, messages, chatTemplate, jsonSchema, tools, parallelToolCalls, toolChoice)
            } catch (e: Exception) {
                null
            }
        }
    }
    
    fun rewind() {
        handle?.let { h ->
            CactusContext.rewind(h)
        }
    }
    
    fun getNCtx(): Int {
        return handle?.let { h ->
            try {
                CactusContext.getNCtx(h)
            } catch (e: Exception) {
                0
            }
        } ?: 0
    }
    
    fun getNEmbd(): Int {
        return handle?.let { h ->
            try {
                CactusContext.getNEmbd(h)
            } catch (e: Exception) {
                0
            }
        } ?: 0
    }
    
    fun getModelDesc(): String {
        return handle?.let { h ->
            try {
                CactusContext.getModelDesc(h)
            } catch (e: Exception) {
                ""
            }
        } ?: ""
    }
    
    fun getModelSize(): Long {
        return handle?.let { h ->
            try {
                CactusContext.getModelSize(h)
            } catch (e: Exception) {
                0L
            }
        } ?: 0L
    }
    
    fun getModelParams(): Long {
        return handle?.let { h ->
            try {
                CactusContext.getModelParams(h)
            } catch (e: Exception) {
                0L
            }
        } ?: 0L
    }
    
    fun isInitialized(): Boolean = handle != null
} 