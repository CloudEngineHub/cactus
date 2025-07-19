package com.cactus

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import platform.Foundation.*

@OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
actual suspend fun downloadSTTModel(url: String, filename: String): Boolean {
    return withContext(Dispatchers.Default) {
        try {
            val documentsDir = NSSearchPathForDirectoriesInDomains(
                NSDocumentDirectory, NSUserDomainMask, true
            ).firstOrNull() as? String ?: return@withContext false
            
            val modelsDir = "$documentsDir/models/vosk"
            NSFileManager.defaultManager.createDirectoryAtPath(
                modelsDir, true, null, null
            )
            
            val modelPath = "$modelsDir/$filename"
            if (NSFileManager.defaultManager.fileExistsAtPath(modelPath)) {
                return@withContext true
            }
            
            val nsUrl = NSURL(string = url) ?: return@withContext false
            val data = NSData.dataWithContentsOfURL(nsUrl) ?: return@withContext false
            
            data.writeToFile(modelPath, true)
        } catch (e: Exception) {
            false
        }
    }
}

actual suspend fun initializeSTT(): Boolean {
    return initializeSpeechRecognition()
}

actual suspend fun performSTT(language: String, maxDuration: Int): STTResult? {
    return try {
        val params = SpeechRecognitionParams(
            language = language,
            enablePartialResults = false,
            maxDuration = maxDuration
        )
        val speechResult = performSpeechRecognition(params)
        speechResult?.let {
            STTResult(
                text = it.text,
                confidence = it.confidence,
                isPartial = it.isPartial
            )
        }
    } catch (e: Exception) {
        null
    }
}

actual suspend fun performFileSTT(audioPath: String, language: String): STTResult? {
    return try {
        val params = SpeechRecognitionParams(
            language = language,
            enablePartialResults = false,
            maxDuration = 60
        )
        val speechResult = recognizeSpeechFromFile(audioPath, params)
        speechResult?.let {
            STTResult(
                text = it.text,
                confidence = it.confidence,
                isPartial = it.isPartial
            )
        }
    } catch (e: Exception) {
        null
    }
}

actual fun stopSTT() {
    stopSpeechRecognition()
} 