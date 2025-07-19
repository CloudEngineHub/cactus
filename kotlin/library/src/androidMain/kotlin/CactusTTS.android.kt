package com.cactus

import android.content.Context
import android.speech.tts.TextToSpeech
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.net.URL
import java.util.*

private val applicationContext: Context by lazy {
    CactusContextInitializer.getApplicationContext()
}

actual suspend fun downloadTTSModel(url: String, filename: String): Boolean {
    return withContext(Dispatchers.IO) {
        try {
            val modelsDir = File(applicationContext.cacheDir, "models/tts")
            if (!modelsDir.exists()) modelsDir.mkdirs()
            
            val modelFile = File(modelsDir, filename)
            if (modelFile.exists()) return@withContext true
            
            val urlConnection = URL(url).openConnection()
            urlConnection.getInputStream().use { input ->
                modelFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            modelFile.exists()
        } catch (e: Exception) {
            false
        }
    }
}

actual suspend fun initializeTTS(): Boolean {
    return withContext(Dispatchers.Default) {
        try {
            val tts = TextToSpeech(applicationContext) { status ->
                // TTS initialization complete
            }
            true
        } catch (e: Exception) {
            false
        }
    }
}

actual suspend fun performTTS(text: String, outputPath: String?, voice: String, speed: Float, pitch: Float, sampleRate: Int): TTSResult? {
    return withContext(Dispatchers.Default) {
        try {
            val tts = TextToSpeech(applicationContext, null)
            
            if (outputPath != null) {
                val file = File(outputPath)
                tts.synthesizeToFile(text, null, file, "tts_synthesis")
            }
            
            TTSResult(
                audioPath = outputPath ?: "",
                duration = 0f,
                success = true
            )
        } catch (e: Exception) {
            null
        }
    }
}

actual suspend fun speakTTS(text: String, voice: String, speed: Float, pitch: Float): Boolean {
    return withContext(Dispatchers.Default) {
        try {
            val tts = TextToSpeech(applicationContext, null)
            tts.setSpeechRate(speed)
            tts.setPitch(pitch)
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "tts_speak")
            true
        } catch (e: Exception) {
            false
        }
    }
}

actual fun stopTTS() {
    try {
        // Stop TTS if needed
    } catch (e: Exception) {
    }
} 