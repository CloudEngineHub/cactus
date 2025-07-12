package com.cactus

import com.sun.jna.Library
import com.sun.jna.Native
import com.sun.jna.Pointer
import com.sun.jna.Structure
import com.sun.jna.ptr.PointerByReference

@Structure.FieldOrder(
    "model_path", "chat_template", "n_ctx", "n_batch", "n_ubatch", "n_gpu_layers", 
    "n_threads", "use_mmap", "use_mlock", "embedding", "pooling_type", "embd_normalize",
    "flash_attn", "cache_type_k", "cache_type_v", "progress_callback"
)
internal class CactusInitParamsC : Structure() {
    @JvmField var model_path: String? = null
    @JvmField var chat_template: String? = null
    @JvmField var n_ctx: Int = 0
    @JvmField var n_batch: Int = 0
    @JvmField var n_ubatch: Int = 0
    @JvmField var n_gpu_layers: Int = 0
    @JvmField var n_threads: Int = 0
    @JvmField var use_mmap: Boolean = false
    @JvmField var use_mlock: Boolean = false
    @JvmField var embedding: Boolean = false
    @JvmField var pooling_type: Int = 0
    @JvmField var embd_normalize: Int = 0
    @JvmField var flash_attn: Boolean = false
    @JvmField var cache_type_k: String? = null
    @JvmField var cache_type_v: String? = null
    @JvmField var progress_callback: Pointer? = null
}

@Structure.FieldOrder(
    "prompt", "n_predict", "n_threads", "seed", "temperature", "top_k", "top_p", 
    "min_p", "typical_p", "penalty_last_n", "penalty_repeat", "penalty_freq", 
    "penalty_present", "mirostat", "mirostat_tau", "mirostat_eta", "ignore_eos", 
    "n_probs", "stop_sequences", "stop_sequence_count", "grammar", "token_callback"
)
internal class CactusCompletionParamsC : Structure() {
    @JvmField var prompt: String? = null
    @JvmField var n_predict: Int = 0
    @JvmField var n_threads: Int = 0
    @JvmField var seed: Int = 0
    @JvmField var temperature: Double = 0.0
    @JvmField var top_k: Int = 0
    @JvmField var top_p: Double = 0.0
    @JvmField var min_p: Double = 0.0
    @JvmField var typical_p: Double = 0.0
    @JvmField var penalty_last_n: Int = 0
    @JvmField var penalty_repeat: Double = 0.0
    @JvmField var penalty_freq: Double = 0.0
    @JvmField var penalty_present: Double = 0.0
    @JvmField var mirostat: Int = 0
    @JvmField var mirostat_tau: Double = 0.0
    @JvmField var mirostat_eta: Double = 0.0
    @JvmField var ignore_eos: Boolean = false
    @JvmField var n_probs: Int = 0
    @JvmField var stop_sequences: Pointer? = null
    @JvmField var stop_sequence_count: Int = 0
    @JvmField var grammar: String? = null
    @JvmField var token_callback: Pointer? = null
}

@Structure.FieldOrder(
    "text", "tokens_predicted", "tokens_evaluated", "truncated", "stopped_eos", 
    "stopped_word", "stopped_limit", "stopping_word"
)
internal class CactusCompletionResultC : Structure() {
    @JvmField var text: String? = null
    @JvmField var tokens_predicted: Int = 0
    @JvmField var tokens_evaluated: Int = 0
    @JvmField var truncated: Boolean = false
    @JvmField var stopped_eos: Boolean = false
    @JvmField var stopped_word: Boolean = false
    @JvmField var stopped_limit: Boolean = false
    @JvmField var stopping_word: String? = null
}

@Structure.FieldOrder("tokens", "count")
internal class CactusTokenArrayC : Structure() {
    @JvmField var tokens: Pointer? = null
    @JvmField var count: Int = 0
}

@Structure.FieldOrder("values", "count")
internal class CactusFloatArrayC : Structure() {
    @JvmField var values: Pointer? = null
    @JvmField var count: Int = 0
}

internal interface CactusLibrary : Library {
    fun cactus_init_context_c(params: CactusInitParamsC): Pointer?
    fun cactus_free_context_c(handle: Pointer)
    fun cactus_completion_c(handle: Pointer, params: CactusCompletionParamsC, result: CactusCompletionResultC): Int
    fun cactus_multimodal_completion_c(handle: Pointer, params: CactusCompletionParamsC, mediaPaths: Array<String>, mediaCount: Int, result: CactusCompletionResultC): Int
    fun cactus_stop_completion_c(handle: Pointer)
    fun cactus_tokenize_c(handle: Pointer, text: String): CactusTokenArrayC
    fun cactus_detokenize_c(handle: Pointer, tokens: IntArray, count: Int): String
    fun cactus_embedding_c(handle: Pointer, text: String): CactusFloatArrayC
    fun cactus_free_string_c(str: String)
    fun cactus_free_token_array_c(arr: CactusTokenArrayC)
    fun cactus_free_float_array_c(arr: CactusFloatArrayC)
    fun cactus_free_completion_result_members_c(result: CactusCompletionResultC)
    fun cactus_set_guide_tokens_c(handle: Pointer, tokens: IntArray, count: Int)
    fun cactus_init_multimodal_c(handle: Pointer, mmprojPath: String, useGpu: Boolean): Int
    fun cactus_is_multimodal_enabled_c(handle: Pointer): Boolean
    fun cactus_supports_vision_c(handle: Pointer): Boolean
    fun cactus_supports_audio_c(handle: Pointer): Boolean
    fun cactus_release_multimodal_c(handle: Pointer)
    fun cactus_init_vocoder_c(handle: Pointer, vocoderModelPath: String): Int
    fun cactus_is_vocoder_enabled_c(handle: Pointer): Boolean
    fun cactus_get_tts_type_c(handle: Pointer): Int
    fun cactus_get_formatted_audio_completion_c(handle: Pointer, speakerJson: String?, textToSpeak: String): String
    fun cactus_get_audio_guide_tokens_c(handle: Pointer, textToSpeak: String): CactusTokenArrayC
    fun cactus_decode_audio_tokens_c(handle: Pointer, tokens: IntArray, count: Int): CactusFloatArrayC
    fun cactus_release_vocoder_c(handle: Pointer)
    fun cactus_rewind_c(handle: Pointer)
    fun cactus_init_sampling_c(handle: Pointer): Boolean
    fun cactus_begin_completion_c(handle: Pointer)
    fun cactus_end_completion_c(handle: Pointer)
    fun cactus_load_prompt_c(handle: Pointer)
    fun cactus_load_prompt_with_media_c(handle: Pointer, mediaPaths: Array<String>, mediaCount: Int)
    fun cactus_do_completion_step_c(handle: Pointer, tokenText: PointerByReference): Int
    fun cactus_find_stopping_strings_c(handle: Pointer, text: String, lastTokenSize: Int, stopType: Int): Long
    fun cactus_get_n_ctx_c(handle: Pointer): Int
    fun cactus_get_n_embd_c(handle: Pointer): Int
    fun cactus_get_model_desc_c(handle: Pointer): String
    fun cactus_get_model_size_c(handle: Pointer): Long
    fun cactus_get_model_params_c(handle: Pointer): Long
    
    companion object {
        val INSTANCE: CactusLibrary = Native.load("cactus", CactusLibrary::class.java)
    }
}

actual object CactusContext {
    private val lib = CactusLibrary.INSTANCE
    
    actual fun initContext(params: CactusInitParams): CactusContextHandle {
        val cParams = CactusInitParamsC().apply {
            model_path = params.modelPath
            chat_template = params.chatTemplate
            n_ctx = params.nCtx
            n_batch = params.nBatch
            n_ubatch = params.nUbatch
            n_gpu_layers = params.nGpuLayers
            n_threads = params.nThreads
            use_mmap = params.useMmap
            use_mlock = params.useMlock
            embedding = params.embedding
            pooling_type = params.poolingType
            embd_normalize = params.embdNormalize
            flash_attn = params.flashAttn
            cache_type_k = params.cacheTypeK
            cache_type_v = params.cacheTypeV
        }
        val handle = lib.cactus_init_context_c(cParams)
        return handle?.let { Pointer.nativeValue(it) } ?: 0L
    }
    
    actual fun freeContext(handle: CactusContextHandle) {
        if (handle != 0L) {
            lib.cactus_free_context_c(Pointer(handle))
        }
    }
    
    actual fun completion(handle: CactusContextHandle, params: CactusCompletionParams): CactusCompletionResult {
        val cParams = CactusCompletionParamsC().apply {
            prompt = params.prompt
            n_predict = params.nPredict
            n_threads = params.nThreads
            seed = params.seed
            temperature = params.temperature
            top_k = params.topK
            top_p = params.topP
            min_p = params.minP
            typical_p = params.typicalP
            penalty_last_n = params.penaltyLastN
            penalty_repeat = params.penaltyRepeat
            penalty_freq = params.penaltyFreq
            penalty_present = params.penaltyPresent
            mirostat = params.mirostat
            mirostat_tau = params.mirostatTau
            mirostat_eta = params.mirostatEta
            ignore_eos = params.ignoreEos
            n_probs = params.nProbs
            grammar = params.grammar
        }
        val result = CactusCompletionResultC()
        lib.cactus_completion_c(Pointer(handle), cParams, result)
        
        return CactusCompletionResult(
            text = result.text ?: "",
            tokensPredicted = result.tokens_predicted,
            tokensEvaluated = result.tokens_evaluated,
            truncated = result.truncated,
            stoppedEos = result.stopped_eos,
            stoppedWord = result.stopped_word,
            stoppedLimit = result.stopped_limit,
            stoppingWord = result.stopping_word
        ).also {
            lib.cactus_free_completion_result_members_c(result)
        }
    }
    
    actual fun multimodalCompletion(handle: CactusContextHandle, params: CactusCompletionParams, mediaPaths: List<String>): CactusCompletionResult {
        val cParams = CactusCompletionParamsC().apply {
            prompt = params.prompt
            n_predict = params.nPredict
            n_threads = params.nThreads
            seed = params.seed
            temperature = params.temperature
            top_k = params.topK
            top_p = params.topP
            min_p = params.minP
            typical_p = params.typicalP
            penalty_last_n = params.penaltyLastN
            penalty_repeat = params.penaltyRepeat
            penalty_freq = params.penaltyFreq
            penalty_present = params.penaltyPresent
            mirostat = params.mirostat
            mirostat_tau = params.mirostatTau
            mirostat_eta = params.mirostatEta
            ignore_eos = params.ignoreEos
            n_probs = params.nProbs
            grammar = params.grammar
        }
        val result = CactusCompletionResultC()
        lib.cactus_multimodal_completion_c(Pointer(handle), cParams, mediaPaths.toTypedArray(), mediaPaths.size, result)
        
        return CactusCompletionResult(
            text = result.text ?: "",
            tokensPredicted = result.tokens_predicted,
            tokensEvaluated = result.tokens_evaluated,
            truncated = result.truncated,
            stoppedEos = result.stopped_eos,
            stoppedWord = result.stopped_word,
            stoppedLimit = result.stopped_limit,
            stoppingWord = result.stopping_word
        ).also {
            lib.cactus_free_completion_result_members_c(result)
        }
    }
    
    actual fun stopCompletion(handle: CactusContextHandle) {
        lib.cactus_stop_completion_c(Pointer(handle))
    }
    
    actual fun tokenize(handle: CactusContextHandle, text: String): CactusTokenArray {
        val result = lib.cactus_tokenize_c(Pointer(handle), text)
        val tokens = result.tokens?.getIntArray(0, result.count) ?: intArrayOf()
        lib.cactus_free_token_array_c(result)
        return CactusTokenArray(tokens, result.count)
    }
    
    actual fun detokenize(handle: CactusContextHandle, tokens: IntArray): String {
        return lib.cactus_detokenize_c(Pointer(handle), tokens, tokens.size)
    }
    
    actual fun tokenizeWithMedia(handle: CactusContextHandle, text: String, mediaPaths: List<String>): CactusTokenizeResult {
        val tokenArray = tokenize(handle, text)
        return CactusTokenizeResult(
            tokens = tokenArray,
            hasMedia = mediaPaths.isNotEmpty(),
            bitmapHashes = emptyList(),
            chunkPositions = emptyList(),
            chunkPositionsMedia = emptyList()
        )
    }
    
    actual fun embedding(handle: CactusContextHandle, text: String): CactusFloatArray {
        val result = lib.cactus_embedding_c(Pointer(handle), text)
        val values = result.values?.getFloatArray(0, result.count) ?: floatArrayOf()
        lib.cactus_free_float_array_c(result)
        return CactusFloatArray(values, result.count)
    }
    
    actual fun setGuideTokens(handle: CactusContextHandle, tokens: IntArray) {
        lib.cactus_set_guide_tokens_c(Pointer(handle), tokens, tokens.size)
    }
    
    actual fun initMultimodal(handle: CactusContextHandle, mmprojPath: String, useGpu: Boolean): Int {
        return lib.cactus_init_multimodal_c(Pointer(handle), mmprojPath, useGpu)
    }
    
    actual fun isMultimodalEnabled(handle: CactusContextHandle): Boolean {
        return lib.cactus_is_multimodal_enabled_c(Pointer(handle))
    }
    
    actual fun supportsVision(handle: CactusContextHandle): Boolean {
        return lib.cactus_supports_vision_c(Pointer(handle))
    }
    
    actual fun supportsAudio(handle: CactusContextHandle): Boolean {
        return lib.cactus_supports_audio_c(Pointer(handle))
    }
    
    actual fun releaseMultimodal(handle: CactusContextHandle) {
        lib.cactus_release_multimodal_c(Pointer(handle))
    }
    
    actual fun initVocoder(handle: CactusContextHandle, vocoderModelPath: String): Int {
        return lib.cactus_init_vocoder_c(Pointer(handle), vocoderModelPath)
    }
    
    actual fun isVocoderEnabled(handle: CactusContextHandle): Boolean {
        return lib.cactus_is_vocoder_enabled_c(Pointer(handle))
    }
    
    actual fun getTTSType(handle: CactusContextHandle): Int {
        return lib.cactus_get_tts_type_c(Pointer(handle))
    }
    
    actual fun getFormattedAudioCompletion(handle: CactusContextHandle, speakerJson: String?, textToSpeak: String): String {
        return lib.cactus_get_formatted_audio_completion_c(Pointer(handle), speakerJson, textToSpeak)
    }
    
    actual fun getAudioGuideTokens(handle: CactusContextHandle, textToSpeak: String): CactusTokenArray {
        val result = lib.cactus_get_audio_guide_tokens_c(Pointer(handle), textToSpeak)
        val tokens = result.tokens?.getIntArray(0, result.count) ?: intArrayOf()
        lib.cactus_free_token_array_c(result)
        return CactusTokenArray(tokens, result.count)
    }
    
    actual fun decodeAudioTokens(handle: CactusContextHandle, tokens: IntArray): CactusFloatArray {
        val result = lib.cactus_decode_audio_tokens_c(Pointer(handle), tokens, tokens.size)
        val values = result.values?.getFloatArray(0, result.count) ?: floatArrayOf()
        lib.cactus_free_float_array_c(result)
        return CactusFloatArray(values, result.count)
    }
    
    actual fun releaseVocoder(handle: CactusContextHandle) {
        lib.cactus_release_vocoder_c(Pointer(handle))
    }
    
    actual fun bench(handle: CactusContextHandle, pp: Int, tg: Int, pl: Int, nr: Int): CactusBenchResult {
        return CactusBenchResult(
            modelName = getModelDesc(handle),
            modelSize = getModelSize(handle),
            modelParams = getModelParams(handle),
            ppAvg = 0.0,
            ppStd = 0.0,
            tgAvg = 0.0,
            tgStd = 0.0
        )
    }
    
    actual fun applyLoraAdapters(handle: CactusContextHandle, adapters: List<CactusLoraAdapter>): Int {
        return 0
    }
    
    actual fun removeLoraAdapters(handle: CactusContextHandle) {
    }
    
    actual fun getLoadedLoraAdapters(handle: CactusContextHandle): List<CactusLoraAdapter> {
        return emptyList()
    }
    
    actual fun validateChatTemplate(handle: CactusContextHandle, useJinja: Boolean, name: String?): Boolean {
        return true
    }
    
    actual fun getFormattedChat(handle: CactusContextHandle, messages: String, chatTemplate: String?): String {
        return messages
    }
    
    actual fun getFormattedChatWithJinja(handle: CactusContextHandle, messages: String, chatTemplate: String?, jsonSchema: String?, tools: String?, parallelToolCalls: Boolean, toolChoice: String?): CactusChatResult {
        return CactusChatResult(
            prompt = messages,
            jsonSchema = jsonSchema,
            tools = tools,
            toolChoice = toolChoice,
            parallelToolCalls = parallelToolCalls
        )
    }
    
    actual fun rewind(handle: CactusContextHandle) {
        lib.cactus_rewind_c(Pointer(handle))
    }
    
    actual fun initSampling(handle: CactusContextHandle): Boolean {
        return lib.cactus_init_sampling_c(Pointer(handle))
    }
    
    actual fun beginCompletion(handle: CactusContextHandle) {
        lib.cactus_begin_completion_c(Pointer(handle))
    }
    
    actual fun endCompletion(handle: CactusContextHandle) {
        lib.cactus_end_completion_c(Pointer(handle))
    }
    
    actual fun loadPrompt(handle: CactusContextHandle) {
        lib.cactus_load_prompt_c(Pointer(handle))
    }
    
    actual fun loadPromptWithMedia(handle: CactusContextHandle, mediaPaths: List<String>) {
        lib.cactus_load_prompt_with_media_c(Pointer(handle), mediaPaths.toTypedArray(), mediaPaths.size)
    }
    
    actual fun doCompletionStep(handle: CactusContextHandle): Pair<Int, String> {
        val tokenTextRef = PointerByReference()
        val tokenId = lib.cactus_do_completion_step_c(Pointer(handle), tokenTextRef)
        val tokenText = tokenTextRef.value?.getString(0) ?: ""
        return Pair(tokenId, tokenText)
    }
    
    actual fun findStoppingStrings(handle: CactusContextHandle, text: String, lastTokenSize: Int, stopType: Int): Long {
        return lib.cactus_find_stopping_strings_c(Pointer(handle), text, lastTokenSize, stopType)
    }
    
    actual fun getNCtx(handle: CactusContextHandle): Int {
        return lib.cactus_get_n_ctx_c(Pointer(handle))
    }
    
    actual fun getNEmbd(handle: CactusContextHandle): Int {
        return lib.cactus_get_n_embd_c(Pointer(handle))
    }
    
    actual fun getModelDesc(handle: CactusContextHandle): String {
        return lib.cactus_get_model_desc_c(Pointer(handle))
    }
    
    actual fun getModelSize(handle: CactusContextHandle): Long {
        return lib.cactus_get_model_size_c(Pointer(handle))
    }
    
    actual fun getModelParams(handle: CactusContextHandle): Long {
        return lib.cactus_get_model_params_c(Pointer(handle))
    }
} 