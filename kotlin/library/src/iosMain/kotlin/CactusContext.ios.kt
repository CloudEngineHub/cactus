package com.cactus

import com.cactus.native.*
import kotlinx.cinterop.*
import platform.Foundation.*

actual object CactusContext {
    
    actual fun initContext(params: CactusInitParams): CactusContextHandle {
        return memScoped {
            val cParams = alloc<cactus_init_params_c_t>().apply {
                model_path = params.modelPath.cstr.ptr
                chat_template = params.chatTemplate?.cstr?.ptr
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
                cache_type_k = params.cacheTypeK?.cstr?.ptr
                cache_type_v = params.cacheTypeV?.cstr?.ptr
                progress_callback = null
            }
            val handle = cactus_init_context_c(cParams.ptr)
            handle?.rawValue ?: 0L
        }
    }
    
    actual fun freeContext(handle: CactusContextHandle) {
        if (handle != 0L) {
            cactus_free_context_c(interpretCPointer(handle))
        }
    }
    
    actual fun completion(handle: CactusContextHandle, params: CactusCompletionParams): CactusCompletionResult {
        return memScoped {
            val cParams = alloc<cactus_completion_params_c_t>().apply {
                prompt = params.prompt.cstr.ptr
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
                stop_sequences = null
                stop_sequence_count = 0
                grammar = params.grammar?.cstr?.ptr
                token_callback = null
            }
            
            val result = alloc<cactus_completion_result_c_t>()
            cactus_completion_c(interpretCPointer(handle), cParams.ptr, result.ptr)
            
            val completionResult = CactusCompletionResult(
                text = result.text?.toKString() ?: "",
                tokensPredicted = result.tokens_predicted,
                tokensEvaluated = result.tokens_evaluated,
                truncated = result.truncated,
                stoppedEos = result.stopped_eos,
                stoppedWord = result.stopped_word,
                stoppedLimit = result.stopped_limit,
                stoppingWord = result.stopping_word?.toKString()
            )
            
            cactus_free_completion_result_members_c(result.ptr)
            completionResult
        }
    }
    
    actual fun multimodalCompletion(handle: CactusContextHandle, params: CactusCompletionParams, mediaPaths: List<String>): CactusCompletionResult {
        return memScoped {
            val cParams = alloc<cactus_completion_params_c_t>().apply {
                prompt = params.prompt.cstr.ptr
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
                stop_sequences = null
                stop_sequence_count = 0
                grammar = params.grammar?.cstr?.ptr
                token_callback = null
            }
            
            val mediaPathsArray = allocArray<CPointerVar<ByteVar>>(mediaPaths.size)
            mediaPaths.forEachIndexed { index, path ->
                mediaPathsArray[index] = path.cstr.ptr
            }
            
            val result = alloc<cactus_completion_result_c_t>()
            cactus_multimodal_completion_c(
                interpretCPointer(handle), 
                cParams.ptr, 
                mediaPathsArray, 
                mediaPaths.size, 
                result.ptr
            )
            
            val completionResult = CactusCompletionResult(
                text = result.text?.toKString() ?: "",
                tokensPredicted = result.tokens_predicted,
                tokensEvaluated = result.tokens_evaluated,
                truncated = result.truncated,
                stoppedEos = result.stopped_eos,
                stoppedWord = result.stopped_word,
                stoppedLimit = result.stopped_limit,
                stoppingWord = result.stopping_word?.toKString()
            )
            
            cactus_free_completion_result_members_c(result.ptr)
            completionResult
        }
    }
    
    actual fun stopCompletion(handle: CactusContextHandle) {
        cactus_stop_completion_c(interpretCPointer(handle))
    }
    
    actual fun tokenize(handle: CactusContextHandle, text: String): CactusTokenArray {
        return memScoped {
            val result = cactus_tokenize_c(interpretCPointer(handle), text.cstr)
            val tokens = IntArray(result.count) { index ->
                result.tokens!![index]
            }
            cactus_free_token_array_c(result)
            CactusTokenArray(tokens, result.count)
        }
    }
    
    actual fun detokenize(handle: CactusContextHandle, tokens: IntArray): String {
        return memScoped {
            val tokensPtr = allocArray<IntVar>(tokens.size)
            tokens.forEachIndexed { index, token ->
                tokensPtr[index] = token
            }
            val result = cactus_detokenize_c(interpretCPointer(handle), tokensPtr, tokens.size)
            result?.toKString() ?: ""
        }
    }
    
    actual fun tokenizeWithMedia(handle: CactusContextHandle, text: String, mediaPaths: List<String>): CactusTokenizeResult {
        return memScoped {
            val mediaPathsArray = allocArray<CPointerVar<ByteVar>>(mediaPaths.size)
            mediaPaths.forEachIndexed { index, path ->
                mediaPathsArray[index] = path.cstr.ptr
            }
            
            val result = cactus_tokenize_with_media_c(
                interpretCPointer(handle), 
                text.cstr, 
                mediaPathsArray, 
                mediaPaths.size
            )
            
            val tokens = IntArray(result.tokens.count) { index ->
                result.tokens.tokens!![index]
            }
            
            val bitmapHashes = mutableListOf<String>()
            for (i in 0 until result.bitmap_hash_count) {
                result.bitmap_hashes!![i]?.toKString()?.let { bitmapHashes.add(it) }
            }
            
            val chunkPositions = mutableListOf<Long>()
            for (i in 0 until result.chunk_position_count) {
                chunkPositions.add(result.chunk_positions!![i].toLong())
            }
            
            val chunkPositionsMedia = mutableListOf<Long>()
            for (i in 0 until result.chunk_position_media_count) {
                chunkPositionsMedia.add(result.chunk_positions_media!![i].toLong())
            }
            
            val tokenizeResult = CactusTokenizeResult(
                tokens = CactusTokenArray(tokens, result.tokens.count),
                hasMedia = result.has_media,
                bitmapHashes = bitmapHashes,
                chunkPositions = chunkPositions,
                chunkPositionsMedia = chunkPositionsMedia
            )
            
            cactus_free_tokenize_result_c(result.ptr)
            tokenizeResult
        }
    }
    
    actual fun embedding(handle: CactusContextHandle, text: String): CactusFloatArray {
        return memScoped {
            val result = cactus_embedding_c(interpretCPointer(handle), text.cstr)
            val values = FloatArray(result.count) { index ->
                result.values!![index]
            }
            cactus_free_float_array_c(result)
            CactusFloatArray(values, result.count)
        }
    }
    
    actual fun setGuideTokens(handle: CactusContextHandle, tokens: IntArray) {
        memScoped {
            val tokensPtr = allocArray<IntVar>(tokens.size)
            tokens.forEachIndexed { index, token ->
                tokensPtr[index] = token
            }
            cactus_set_guide_tokens_c(interpretCPointer(handle), tokensPtr, tokens.size)
        }
    }
    
    actual fun initMultimodal(handle: CactusContextHandle, mmprojPath: String, useGpu: Boolean): Int {
        return cactus_init_multimodal_c(interpretCPointer(handle), mmprojPath.cstr, useGpu)
    }
    
    actual fun isMultimodalEnabled(handle: CactusContextHandle): Boolean {
        return cactus_is_multimodal_enabled_c(interpretCPointer(handle))
    }
    
    actual fun supportsVision(handle: CactusContextHandle): Boolean {
        return cactus_supports_vision_c(interpretCPointer(handle))
    }
    
    actual fun supportsAudio(handle: CactusContextHandle): Boolean {
        return cactus_supports_audio_c(interpretCPointer(handle))
    }
    
    actual fun releaseMultimodal(handle: CactusContextHandle) {
        cactus_release_multimodal_c(interpretCPointer(handle))
    }
    
    actual fun initVocoder(handle: CactusContextHandle, vocoderModelPath: String): Int {
        return cactus_init_vocoder_c(interpretCPointer(handle), vocoderModelPath.cstr)
    }
    
    actual fun isVocoderEnabled(handle: CactusContextHandle): Boolean {
        return cactus_is_vocoder_enabled_c(interpretCPointer(handle))
    }
    
    actual fun getTTSType(handle: CactusContextHandle): Int {
        return cactus_get_tts_type_c(interpretCPointer(handle))
    }
    
    actual fun getFormattedAudioCompletion(handle: CactusContextHandle, speakerJson: String?, textToSpeak: String): String {
        val result = cactus_get_formatted_audio_completion_c(
            interpretCPointer(handle), 
            speakerJson?.cstr, 
            textToSpeak.cstr
        )
        return result?.toKString() ?: ""
    }
    
    actual fun getAudioGuideTokens(handle: CactusContextHandle, textToSpeak: String): CactusTokenArray {
        return memScoped {
            val result = cactus_get_audio_guide_tokens_c(interpretCPointer(handle), textToSpeak.cstr)
            val tokens = IntArray(result.count) { index ->
                result.tokens!![index]
            }
            cactus_free_token_array_c(result)
            CactusTokenArray(tokens, result.count)
        }
    }
    
    actual fun decodeAudioTokens(handle: CactusContextHandle, tokens: IntArray): CactusFloatArray {
        return memScoped {
            val tokensPtr = allocArray<IntVar>(tokens.size)
            tokens.forEachIndexed { index, token ->
                tokensPtr[index] = token
            }
            val result = cactus_decode_audio_tokens_c(interpretCPointer(handle), tokensPtr, tokens.size)
            val values = FloatArray(result.count) { index ->
                result.values!![index]
            }
            cactus_free_float_array_c(result)
            CactusFloatArray(values, result.count)
        }
    }
    
    actual fun releaseVocoder(handle: CactusContextHandle) {
        cactus_release_vocoder_c(interpretCPointer(handle))
    }
    
    actual fun bench(handle: CactusContextHandle, pp: Int, tg: Int, pl: Int, nr: Int): CactusBenchResult {
        return memScoped {
            val result = cactus_bench_c(interpretCPointer(handle), pp, tg, pl, nr)
            val benchResult = CactusBenchResult(
                modelName = result.model_name?.toKString() ?: "",
                modelSize = result.model_size,
                modelParams = result.model_params,
                ppAvg = result.pp_avg,
                ppStd = result.pp_std,
                tgAvg = result.tg_avg,
                tgStd = result.tg_std
            )
            cactus_free_bench_result_members_c(result.ptr)
            benchResult
        }
    }
    
    actual fun applyLoraAdapters(handle: CactusContextHandle, adapters: List<CactusLoraAdapter>): Int {
        return memScoped {
            val cAdapters = alloc<cactus_lora_adapters_c_t>().apply {
                count = adapters.size
                this.adapters = allocArray<cactus_lora_adapter_c_t>(adapters.size)
                adapters.forEachIndexed { index, adapter ->
                    this.adapters!![index].apply {
                        path = adapter.path.cstr.ptr
                        scale = adapter.scale
                    }
                }
            }
            cactus_apply_lora_adapters_c(interpretCPointer(handle), cAdapters.ptr)
        }
    }
    
    actual fun removeLoraAdapters(handle: CactusContextHandle) {
        cactus_remove_lora_adapters_c(interpretCPointer(handle))
    }
    
    actual fun getLoadedLoraAdapters(handle: CactusContextHandle): List<CactusLoraAdapter> {
        return memScoped {
            val result = cactus_get_loaded_lora_adapters_c(interpretCPointer(handle))
            val adapters = mutableListOf<CactusLoraAdapter>()
            for (i in 0 until result.count) {
                val adapter = result.adapters!![i]
                adapters.add(CactusLoraAdapter(
                    path = adapter.path?.toKString() ?: "",
                    scale = adapter.scale
                ))
            }
            cactus_free_lora_adapters_c(result.ptr)
            adapters
        }
    }
    
    actual fun validateChatTemplate(handle: CactusContextHandle, useJinja: Boolean, name: String?): Boolean {
        return cactus_validate_chat_template_c(interpretCPointer(handle), useJinja, name?.cstr)
    }
    
    actual fun getFormattedChat(handle: CactusContextHandle, messages: String, chatTemplate: String?): String {
        val result = cactus_get_formatted_chat_c(interpretCPointer(handle), messages.cstr, chatTemplate?.cstr)
        return result?.toKString() ?: ""
    }
    
    actual fun getFormattedChatWithJinja(handle: CactusContextHandle, messages: String, chatTemplate: String?, jsonSchema: String?, tools: String?, parallelToolCalls: Boolean, toolChoice: String?): CactusChatResult {
        return memScoped {
            val result = cactus_get_formatted_chat_with_jinja_c(
                interpretCPointer(handle),
                messages.cstr,
                chatTemplate?.cstr,
                jsonSchema?.cstr,
                tools?.cstr,
                parallelToolCalls,
                toolChoice?.cstr
            )
            val chatResult = CactusChatResult(
                prompt = result.prompt?.toKString() ?: "",
                jsonSchema = result.json_schema?.toKString(),
                tools = result.tools?.toKString(),
                toolChoice = result.tool_choice?.toKString(),
                parallelToolCalls = result.parallel_tool_calls
            )
            cactus_free_chat_result_members_c(result.ptr)
            chatResult
        }
    }
    
    actual fun rewind(handle: CactusContextHandle) {
        cactus_rewind_c(interpretCPointer(handle))
    }
    
    actual fun initSampling(handle: CactusContextHandle): Boolean {
        return cactus_init_sampling_c(interpretCPointer(handle))
    }
    
    actual fun beginCompletion(handle: CactusContextHandle) {
        cactus_begin_completion_c(interpretCPointer(handle))
    }
    
    actual fun endCompletion(handle: CactusContextHandle) {
        cactus_end_completion_c(interpretCPointer(handle))
    }
    
    actual fun loadPrompt(handle: CactusContextHandle) {
        cactus_load_prompt_c(interpretCPointer(handle))
    }
    
    actual fun loadPromptWithMedia(handle: CactusContextHandle, mediaPaths: List<String>) {
        memScoped {
            val mediaPathsArray = allocArray<CPointerVar<ByteVar>>(mediaPaths.size)
            mediaPaths.forEachIndexed { index, path ->
                mediaPathsArray[index] = path.cstr.ptr
            }
            cactus_load_prompt_with_media_c(interpretCPointer(handle), mediaPathsArray, mediaPaths.size)
        }
    }
    
    actual fun doCompletionStep(handle: CactusContextHandle): Pair<Int, String> {
        return memScoped {
            val tokenTextPtr = alloc<CPointerVar<ByteVar>>()
            val tokenId = cactus_do_completion_step_c(interpretCPointer(handle), tokenTextPtr.ptr)
            val tokenText = tokenTextPtr.value?.toKString() ?: ""
            Pair(tokenId, tokenText)
        }
    }
    
    actual fun findStoppingStrings(handle: CactusContextHandle, text: String, lastTokenSize: Int, stopType: Int): Long {
        return cactus_find_stopping_strings_c(interpretCPointer(handle), text.cstr, lastTokenSize.toULong(), stopType).toLong()
    }
    
    actual fun getNCtx(handle: CactusContextHandle): Int {
        return cactus_get_n_ctx_c(interpretCPointer(handle))
    }
    
    actual fun getNEmbd(handle: CactusContextHandle): Int {
        return cactus_get_n_embd_c(interpretCPointer(handle))
    }
    
    actual fun getModelDesc(handle: CactusContextHandle): String {
        val result = cactus_get_model_desc_c(interpretCPointer(handle))
        return result?.toKString() ?: ""
    }
    
    actual fun getModelSize(handle: CactusContextHandle): Long {
        return cactus_get_model_size_c(interpretCPointer(handle))
    }
    
    actual fun getModelParams(handle: CactusContextHandle): Long {
        return cactus_get_model_params_c(interpretCPointer(handle))
    }
} 