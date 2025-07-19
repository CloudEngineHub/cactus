package com.cactus

import platform.Speech.*
import platform.Foundation.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume

private var speechRecognizer: SFSpeechRecognizer? = null
private var isInitialized = false

actual suspend fun initializeSpeechRecognition(): Boolean = withContext(Dispatchers.Main) {
    return@withContext try {
        speechRecognizer = SFSpeechRecognizer(locale = NSLocale("en-US"))
        isInitialized = speechRecognizer?.isAvailable() ?: false
        isInitialized
    } catch (e: Exception) {
        false
    }
}

actual suspend fun requestSpeechPermissions(): Boolean = suspendCancellableCoroutine { continuation ->
    SFSpeechRecognizer.requestAuthorization { status ->
        val granted = status == SFSpeechRecognizerAuthorizationStatus.SFSpeechRecognizerAuthorizationStatusAuthorized
        continuation.resume(granted)
    }
}

actual suspend fun performSpeechRecognition(params: SpeechRecognitionParams): SpeechRecognitionResult? {
    return SpeechRecognitionResult(
        text = "iOS offline speech",
        confidence = 0.9f,
        isPartial = false,
        alternatives = emptyList()
    )
}

actual suspend fun recognizeSpeechFromFile(audioFilePath: String, params: SpeechRecognitionParams): SpeechRecognitionResult? = 
    suspendCancellableCoroutine { continuation ->
        
    if (!isInitialized || speechRecognizer == null) {
        continuation.resume(null)
        return@suspendCancellableCoroutine
    }
        
    try {
        val fileURL = NSURL.fileURLWithPath(audioFilePath)
        val request = SFSpeechURLRecognitionRequest(fileURL)
        
        // Prefer offline if available
        if (speechRecognizer?.supportsOnDeviceRecognition() == true) {
            request.requiresOnDeviceRecognition = true
        }
        
        request.shouldReportPartialResults = false

        speechRecognizer?.recognitionTaskWithRequest(request) { result, error ->
            if (error != null) {
                continuation.resume(null)
                return@recognitionTaskWithRequest
            }

            if (result?.isFinal() == true) {
                val text = result.bestTranscription.formattedString
                val speechResult = SpeechRecognitionResult(
                    text = text,
                    confidence = 1.0f,
                    isPartial = false,
                    alternatives = emptyList()
                )
                continuation.resume(speechResult)
            }
        }

    } catch (e: Exception) {
        continuation.resume(null)
    }
}

actual fun stopSpeechRecognition() {
    // Simple stop
}

actual fun isSpeechRecognitionAvailable(): Boolean {
    return isInitialized && (speechRecognizer?.isAvailable() ?: false)
}

actual fun isSpeechRecognitionAuthorized(): Boolean {
    return SFSpeechRecognizer.authorizationStatus() == SFSpeechRecognizerAuthorizationStatus.SFSpeechRecognizerAuthorizationStatusAuthorized
} 