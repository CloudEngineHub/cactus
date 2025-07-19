package com.cactus

import platform.Speech.*
import platform.Foundation.*
import platform.AVFAudio.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.delay
import kotlin.coroutines.resume
import kotlin.coroutines.Continuation
import kotlin.native.ref.WeakReference
import kotlinx.cinterop.ExperimentalForeignApi

private var speechRecognizer: SFSpeechRecognizer? = null
private var audioEngine: AVAudioEngine? = null
private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest? = null
private var recognitionTask: SFSpeechRecognitionTask? = null
private var isInitialized = false
private var isRecording = false

@OptIn(ExperimentalForeignApi::class)
actual suspend fun initializeSpeechRecognition(): Boolean = withContext(Dispatchers.Main) {
    return@withContext try {
        speechRecognizer = SFSpeechRecognizer(locale = NSLocale("en-US"))
        audioEngine = AVAudioEngine()
        
        val audioSession = AVAudioSession.sharedInstance()
        audioSession.setCategory(AVAudioSessionCategoryRecord, null)
        audioSession.setActive(true, null)
        
        isInitialized = speechRecognizer?.isAvailable() ?: false
        isInitialized
    } catch (e: Exception) {
        println("Failed to initialize speech recognition: $e")
        false
    }
}

actual suspend fun requestSpeechPermissions(): Boolean = suspendCancellableCoroutine { continuation ->
    SFSpeechRecognizer.requestAuthorization { status ->
        val granted = status == SFSpeechRecognizerAuthorizationStatus.SFSpeechRecognizerAuthorizationStatusAuthorized
        continuation.resume(granted)
    }
}

@OptIn(ExperimentalForeignApi::class)
actual suspend fun performSpeechRecognition(params: SpeechRecognitionParams): SpeechRecognitionResult? = 
    suspendCancellableCoroutine { continuation ->
        
    if (!isInitialized || speechRecognizer == null || audioEngine == null) {
        println("Speech recognition not initialized")
        continuation.resume(null)
        return@suspendCancellableCoroutine
    }
    
    if (isRecording) {
        println("Already recording")
        continuation.resume(null)
        return@suspendCancellableCoroutine
    }
    
    try {
        stopCurrentRecognition()
        
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        val request = recognitionRequest ?: run {
            continuation.resume(null)
            return@suspendCancellableCoroutine
        }
        
        if (speechRecognizer?.supportsOnDeviceRecognition() == true) {
            request.requiresOnDeviceRecognition = true
        }
        
        request.shouldReportPartialResults = params.enablePartialResults
        
        val inputNode = audioEngine!!.inputNode
        val recordingFormat = inputNode.outputFormatForBus(0u)
        
        inputNode.installTapOnBus(
            bus = 0u,
            bufferSize = 1024u,
            format = recordingFormat
        ) { buffer, _ ->
            recognitionRequest?.appendAudioPCMBuffer(buffer!!)
        }
        
        audioEngine!!.prepare()
        audioEngine!!.startAndReturnError(null)
        isRecording = true
        
        var hasFinalResult = false
        var timeoutTimer: NSTimer? = null
        
        recognitionTask = speechRecognizer!!.recognitionTaskWithRequest(request) { result, error ->
            
            if (error != null) {
                println("Speech recognition error: ${error.localizedDescription}")
                if (!hasFinalResult) {
                    stopCurrentRecognition()
                    continuation.resume(null)
                    timeoutTimer?.invalidate()
                }
                return@recognitionTaskWithRequest
            }
            
            if (result != null) {
                val text = result.bestTranscription.formattedString
                val confidence = 0.9f
                
                if (result.isFinal()) {
                    hasFinalResult = true
                    stopCurrentRecognition()
                    timeoutTimer?.invalidate()
                    
                    val speechResult = SpeechRecognitionResult(
                        text = text,
                        confidence = confidence,
                        isPartial = false,
                        alternatives = emptyList()
                    )
                    continuation.resume(speechResult)
                } else if (params.enablePartialResults && text.isNotEmpty()) {
                    val speechResult = SpeechRecognitionResult(
                        text = text,
                        confidence = confidence,
                        isPartial = true,
                        alternatives = emptyList()
                    )
                }
            }
        }
        
        timeoutTimer = NSTimer.scheduledTimerWithTimeInterval(
            interval = params.maxDuration.toDouble(),
            repeats = false
        ) { _ ->
            if (!hasFinalResult) {
                println("Speech recognition timeout")
                stopCurrentRecognition()
                continuation.resume(SpeechRecognitionResult(
                    text = "",
                    confidence = 0.0f,
                    isPartial = false,
                    alternatives = emptyList()
                ))
            }
        }
        
        continuation.invokeOnCancellation {
            timeoutTimer?.invalidate()
            stopCurrentRecognition()
        }
        
    } catch (e: Exception) {
        println("Failed to start speech recognition: $e")
        stopCurrentRecognition()
        continuation.resume(null)
    }
}

@OptIn(ExperimentalForeignApi::class)
actual suspend fun recognizeSpeechFromFile(audioFilePath: String, params: SpeechRecognitionParams): SpeechRecognitionResult? = 
    suspendCancellableCoroutine { continuation ->
        
    if (!isInitialized || speechRecognizer == null) {
        continuation.resume(null)
        return@suspendCancellableCoroutine
    }
        
    try {
        val fileURL = NSURL.fileURLWithPath(audioFilePath)
        val request = SFSpeechURLRecognitionRequest(fileURL)
        
        if (speechRecognizer?.supportsOnDeviceRecognition() == true) {
            request.requiresOnDeviceRecognition = true
        }
        
        request.shouldReportPartialResults = false

        speechRecognizer?.recognitionTaskWithRequest(request) { result, error ->
            if (error != null) {
                println("File speech recognition error: ${error.localizedDescription}")
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
        println("Failed to recognize speech from file: $e")
        continuation.resume(null)
    }
}

@OptIn(ExperimentalForeignApi::class)
private fun stopCurrentRecognition() {
    if (isRecording) {
        audioEngine?.stop()
        audioEngine?.inputNode?.removeTapOnBus(0u)
        isRecording = false
    }
    
    recognitionRequest?.endAudio()
    recognitionRequest = null
    recognitionTask?.cancel()
    recognitionTask = null
}

actual fun stopSpeechRecognition() {
    stopCurrentRecognition()
}

actual fun isSpeechRecognitionAvailable(): Boolean {
    return isInitialized && (speechRecognizer?.isAvailable() ?: false)
}

actual fun isSpeechRecognitionAuthorized(): Boolean {
    return SFSpeechRecognizer.authorizationStatus() == SFSpeechRecognizerAuthorizationStatus.SFSpeechRecognizerAuthorizationStatusAuthorized
} 