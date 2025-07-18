package com.cactus

import platform.Speech.*
import platform.Foundation.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume

private var speechRecognizer: SFSpeechRecognizer? = null

actual suspend fun initializeSpeechRecognition(): Boolean = withContext(Dispatchers.Main) {
    return@withContext try {
        speechRecognizer = SFSpeechRecognizer(locale = NSLocale("en-US"))
        speechRecognizer?.isAvailable() ?: false
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
    // iOS live speech recognition requires complex audio setup
    // For now, return null - can be implemented later
    return null
}

actual suspend fun recognizeSpeechFromFile(audioFilePath: String, params: SpeechRecognitionParams): SpeechRecognitionResult? = 
    suspendCancellableCoroutine { continuation ->
        
    try {
        val fileURL = NSURL.fileURLWithPath(audioFilePath)
        val request = SFSpeechURLRecognitionRequest(fileURL)
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
    // No complex cleanup needed for simplified implementation
}

actual fun isSpeechRecognitionAvailable(): Boolean {
    return speechRecognizer?.isAvailable() ?: false
}

actual fun isSpeechRecognitionAuthorized(): Boolean {
    return SFSpeechRecognizer.authorizationStatus() == SFSpeechRecognizerAuthorizationStatus.SFSpeechRecognizerAuthorizationStatusAuthorized
} 