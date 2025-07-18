package com.cactus

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.core.content.ContextCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume

private lateinit var applicationContext: Context
private var speechRecognizer: SpeechRecognizer? = null
private var isListening = false

fun initializeAndroidSpeechContext(context: Context) {
    applicationContext = context.applicationContext
}

// Helper function to check if permission is needed for requesting
fun shouldRequestSpeechPermission(): Boolean {
    return ContextCompat.checkSelfPermission(
        applicationContext,
        Manifest.permission.RECORD_AUDIO
    ) != PackageManager.PERMISSION_GRANTED
}

actual suspend fun initializeSpeechRecognition(): Boolean = withContext(Dispatchers.Main) {
    return@withContext try {
        if (SpeechRecognizer.isRecognitionAvailable(applicationContext)) {
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(applicationContext)
            true
        } else {
            false
        }
    } catch (e: Exception) {
        false
    }
}

actual suspend fun requestSpeechPermissions(): Boolean {
    return withContext(Dispatchers.Main) {
        val hasPermission = ContextCompat.checkSelfPermission(
            applicationContext,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
        
        if (!hasPermission) {
            // Log that permission is needed - the app should handle the actual request
            println("Speech permission not granted. App should request RECORD_AUDIO permission.")
        }
        
        hasPermission
    }
}

actual suspend fun performSpeechRecognition(params: SpeechRecognitionParams): SpeechRecognitionResult? = 
    suspendCancellableCoroutine { continuation ->
        
    if (isListening) {
        continuation.resume(null)
        return@suspendCancellableCoroutine
    }

    val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        putExtra(RecognizerIntent.EXTRA_LANGUAGE, params.language)
        putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, params.enablePartialResults)
        putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 5)
        putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, params.maxDuration * 1000L)
    }

    val recognitionListener = object : RecognitionListener {
        override fun onReadyForSpeech(params: Bundle?) {
            isListening = true
        }

        override fun onBeginningOfSpeech() {}
        override fun onRmsChanged(rmsdB: Float) {}
        override fun onBufferReceived(buffer: ByteArray?) {}

        override fun onEndOfSpeech() {
            isListening = false
        }

        override fun onError(error: Int) {
            isListening = false
            continuation.resume(null)
        }

        override fun onResults(results: Bundle?) {
            isListening = false
            val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
            val confidences = results?.getFloatArray(SpeechRecognizer.CONFIDENCE_SCORES)
            
            if (!matches.isNullOrEmpty()) {
                val recognizedText = matches[0]
                val confidence = confidences?.getOrNull(0) ?: 1.0f
                val alternatives = if (matches.size > 1) matches.drop(1) else emptyList()
                
                val result = SpeechRecognitionResult(
                    text = recognizedText,
                    confidence = confidence,
                    isPartial = false,
                    alternatives = alternatives
                )
                continuation.resume(result)
            } else {
                continuation.resume(null)
            }
        }

        override fun onPartialResults(partialResults: Bundle?) {}
        override fun onEvent(eventType: Int, params: Bundle?) {}
    }

    continuation.invokeOnCancellation {
        stopSpeechRecognition()
    }

    try {
        speechRecognizer?.setRecognitionListener(recognitionListener)
        speechRecognizer?.startListening(intent)
    } catch (e: Exception) {
        isListening = false
        continuation.resume(null)
    }
}

actual suspend fun recognizeSpeechFromFile(audioFilePath: String, params: SpeechRecognitionParams): SpeechRecognitionResult? {
    // Android SpeechRecognizer doesn't support file recognition directly
    return null
}

actual fun stopSpeechRecognition() {
    try {
        speechRecognizer?.stopListening()
        isListening = false
    } catch (e: Exception) {
        // Ignore errors when stopping
    }
}

actual fun isSpeechRecognitionAvailable(): Boolean {
    return SpeechRecognizer.isRecognitionAvailable(applicationContext)
}

actual fun isSpeechRecognitionAuthorized(): Boolean {
    return ContextCompat.checkSelfPermission(
        applicationContext,
        Manifest.permission.RECORD_AUDIO
    ) == PackageManager.PERMISSION_GRANTED
} 