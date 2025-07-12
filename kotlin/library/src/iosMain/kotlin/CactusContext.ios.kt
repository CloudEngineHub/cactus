@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
package com.cactus

import kotlinx.cinterop.COpaquePointer

actual object CactusContext {

    actual fun initContext(params: CactusInitParams): CactusContextHandle? = null
    actual fun freeContext(handle: CactusContextHandle) {}

    actual fun completion(handle: CactusContextHandle, params: CactusCompletionParams): CactusCompletionResult =
        CactusCompletionResult("",0,0,false,false,false,false,null)
    actual fun multimodalCompletion(handle: CactusContextHandle, params: CactusCompletionParams, mediaPaths: List<String>): CactusCompletionResult =
        CactusCompletionResult("",0,0,false,false,false,false,null)
    actual fun stopCompletion(handle: CactusContextHandle) {}

    actual fun tokenize(handle: CactusContextHandle, text: String): CactusTokenArray = CactusTokenArray(IntArray(0),0)
    actual fun detokenize(handle: CactusContextHandle, tokens: IntArray): String = ""
    actual fun tokenizeWithMedia(handle: CactusContextHandle, text: String, mediaPaths: List<String>): CactusTokenizeResult =
        CactusTokenizeResult(CactusTokenArray(IntArray(0),0), false, emptyList(), emptyList(), emptyList())

    actual fun embedding(handle: CactusContextHandle, text: String): CactusFloatArray = CactusFloatArray(FloatArray(0),0)
    actual fun setGuideTokens(handle: CactusContextHandle, tokens: IntArray) {}

    actual fun initMultimodal(handle: CactusContextHandle, mmprojPath: String, useGpu: Boolean): Int = -1
    actual fun isMultimodalEnabled(handle: CactusContextHandle): Boolean = false
    actual fun supportsVision(handle: CactusContextHandle): Boolean = false
    actual fun supportsAudio(handle: CactusContextHandle): Boolean = false
    actual fun releaseMultimodal(handle: CactusContextHandle) {}

    actual fun initVocoder(handle: CactusContextHandle, vocoderModelPath: String): Int = -1
    actual fun isVocoderEnabled(handle: CactusContextHandle): Boolean = false
    actual fun getTTSType(handle: CactusContextHandle): Int = -1
    actual fun getFormattedAudioCompletion(handle: CactusContextHandle, speakerJson: String?, textToSpeak: String): String = ""
    actual fun getAudioGuideTokens(handle: CactusContextHandle, textToSpeak: String): CactusTokenArray = CactusTokenArray(IntArray(0),0)
    actual fun decodeAudioTokens(handle: CactusContextHandle, tokens: IntArray): CactusFloatArray = CactusFloatArray(FloatArray(0),0)
    actual fun releaseVocoder(handle: CactusContextHandle) {}

    actual fun bench(handle: CactusContextHandle, pp: Int, tg: Int, pl: Int, nr: Int): CactusBenchResult = CactusBenchResult("",0,0,0.0,0.0,0.0,0.0)
    actual fun applyLoraAdapters(handle: CactusContextHandle, adapters: List<CactusLoraAdapter>): Int = -1
    actual fun removeLoraAdapters(handle: CactusContextHandle) {}
    actual fun getLoadedLoraAdapters(handle: CactusContextHandle): List<CactusLoraAdapter> = emptyList()

    actual fun validateChatTemplate(handle: CactusContextHandle, useJinja: Boolean, name: String?): Boolean = false
    actual fun getFormattedChat(handle: CactusContextHandle, messages: String, chatTemplate: String?): String = messages
    actual fun getFormattedChatWithJinja(handle: CactusContextHandle, messages: String, chatTemplate: String?, jsonSchema: String?, tools: String?, parallelToolCalls: Boolean, toolChoice: String?): CactusChatResult =
        CactusChatResult(messages, jsonSchema, tools, toolChoice, parallelToolCalls)

    actual fun rewind(handle: CactusContextHandle) {}
    actual fun initSampling(handle: CactusContextHandle): Boolean = false
    actual fun beginCompletion(handle: CactusContextHandle) {}
    actual fun endCompletion(handle: CactusContextHandle) {}
    actual fun loadPrompt(handle: CactusContextHandle) {}
    actual fun loadPromptWithMedia(handle: CactusContextHandle, mediaPaths: List<String>) {}
    actual fun doCompletionStep(handle: CactusContextHandle): Pair<Int, String> = 0 to ""
    actual fun findStoppingStrings(handle: CactusContextHandle, text: String, lastTokenSize: Int, stopType: Int): Long = 0L

    actual fun getNCtx(handle: CactusContextHandle): Int = 0
    actual fun getNEmbd(handle: CactusContextHandle): Int = 0
    actual fun getModelDesc(handle: CactusContextHandle): String = ""
    actual fun getModelSize(handle: CactusContextHandle): Long = 0
    actual fun getModelParams(handle: CactusContextHandle): Long = 0
}
