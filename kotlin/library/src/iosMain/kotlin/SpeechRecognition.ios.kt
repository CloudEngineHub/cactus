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
private var lastSpeechTime: Double = 0.0
private var hasDetectedSpeech = false

@OptIn(ExperimentalForeignApi::class)
actual suspend fun initializeSpeechRecognition(): Boolean = withContext(Dispatchers.Main) {
    return@withContext try {
        speechRecognizer = SFSpeechRecognizer(locale = NSLocale("en-US"))
        audioEngine = AVAudioEngine()
        
        val audioSession = AVAudioSession.sharedInstance()
        audioSession.setCategory(AVAudioSessionCategoryRecord, null)
        audioSession.setActive(true, null)
        
        val isAvailable = speechRecognizer?.isAvailable() ?: false
        val supportsOnDevice = speechRecognizer?.supportsOnDeviceRecognition() ?: false
        
        if (!supportsOnDevice) {
            println("On-device speech recognition not supported on this device")
        }
        
        isInitialized = isAvailable
        println("Speech recognition initialized - Available: $isAvailable, On-device: $supportsOnDevice")
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
    
    val supportsOnDevice = speechRecognizer?.supportsOnDeviceRecognition() ?: false
    if (!supportsOnDevice) {
        println("On-device speech recognition not available - device too old or language not supported")
        continuation.resume(SpeechRecognitionResult(
            text = "",
            confidence = 0.0f,
            isPartial = false,
            alternatives = emptyList()
        ))
        return@suspendCancellableCoroutine
    }
    
    try {
        stopCurrentRecognition()
        
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        val request = recognitionRequest ?: run {
            continuation.resume(null)
            return@suspendCancellableCoroutine
        }
        
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = true
        
        println("Starting on-device speech recognition")
        
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
        hasDetectedSpeech = false
        lastSpeechTime = NSDate().timeIntervalSince1970
        
        var hasFinalResult = false
        var timeoutTimer: NSTimer? = null
        var silenceTimer: NSTimer? = null
        var lastTranscriptionLength = 0
        var currentText = ""
        
        println("🎙️ Started listening... (speak now)")
        
        recognitionTask = speechRecognizer!!.recognitionTaskWithRequest(request) { result, error ->
            
            if (error != null) {
                println("Speech recognition error: ${error.localizedDescription}")
                if (!hasFinalResult) {
                    stopCurrentRecognition()
                    continuation.resume(null)
                    timeoutTimer?.invalidate()
                    silenceTimer?.invalidate()
                }
                return@recognitionTaskWithRequest
            }
            
            if (result != null) {
                val text = result.bestTranscription.formattedString
                currentText = text // Update current text
                val confidence = 0.9f
                val currentTime = NSDate().timeIntervalSince1970
                
        
                if (text.length > lastTranscriptionLength && text.trim().isNotEmpty()) {
                    hasDetectedSpeech = true
                    lastSpeechTime = currentTime
                    lastTranscriptionLength = text.length
                    println("Speech detected: '$text'")
                    
                    silenceTimer?.invalidate()
                    
                    if (hasDetectedSpeech) {
                        silenceTimer = NSTimer.scheduledTimerWithTimeInterval(
                            interval = 1.0, 
                            repeats = false
                        ) { _ ->
                            if (!hasFinalResult && hasDetectedSpeech) {
                                println("Silence detected, stopping...")
                                hasFinalResult = true
                                stopCurrentRecognition()
                                timeoutTimer?.invalidate()
                                
                                val speechResult = SpeechRecognitionResult(
                                    text = currentText,
                                    confidence = confidence,
                                    isPartial = false,
                                    alternatives = emptyList()
                                )
                                println("Resuming with silence-detected result: $speechResult")
                                continuation.resume(speechResult)
                            }
                        }
                    }
                }
                
                if (result.isFinal()) {
                    hasFinalResult = true
                    stopCurrentRecognition()
                    timeoutTimer?.invalidate()
                    silenceTimer?.invalidate()
                    
                    val speechResult = SpeechRecognitionResult(
                        text = text,
                        confidence = confidence,
                        isPartial = false,
                        alternatives = emptyList()
                    )
                    println("🔄 Resuming with final result: $speechResult")
                    continuation.resume(speechResult)
                }
            }
        }
        
        // Overall timeout timer
        timeoutTimer = NSTimer.scheduledTimerWithTimeInterval(
            interval = params.maxDuration.toDouble(),
            repeats = false
        ) { _ ->
            if (!hasFinalResult) {
                if (hasDetectedSpeech) {
                    println("Timeout reached, finishing with detected speech")
                    stopCurrentRecognition()
                    silenceTimer?.invalidate()
                    val timeoutResult = SpeechRecognitionResult(
                        text = currentText,
                        confidence = 0.8f,
                        isPartial = false,
                        alternatives = emptyList()
                    )
                    println("Resuming with timeout result (with speech): $timeoutResult")
                    continuation.resume(timeoutResult)
                } else {
                    println("Timeout reached, no speech detected")
                    stopCurrentRecognition()
                    silenceTimer?.invalidate()
                    val noSpeechResult = SpeechRecognitionResult(
                        text = "",
                        confidence = 0.0f,
                        isPartial = false,
                        alternatives = emptyList()
                    )
                    println("Resuming with timeout result (no speech): $noSpeechResult")
                    continuation.resume(noSpeechResult)
                }
            }
        }
        
        silenceTimer = NSTimer.scheduledTimerWithTimeInterval(
            interval = 1.0, 
            repeats = false
        ) { _ ->
            if (!hasFinalResult && !hasDetectedSpeech) {
                println("No speech detected in 1 second, stopping...")
                hasFinalResult = true
                stopCurrentRecognition()
                timeoutTimer?.invalidate()
                val initialSilenceResult = SpeechRecognitionResult(
                    text = "",
                    confidence = 0.0f,
                    isPartial = false,
                    alternatives = emptyList()
                )
                println("Resuming with initial silence result: $initialSilenceResult")
                continuation.resume(initialSilenceResult)
            }
        }
        
        continuation.invokeOnCancellation {
            timeoutTimer?.invalidate()
            silenceTimer?.invalidate()
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
    
    val supportsOnDevice = speechRecognizer?.supportsOnDeviceRecognition() ?: false
    if (!supportsOnDevice) {
        println("On-device speech recognition not available for file processing")
        continuation.resume(SpeechRecognitionResult(
            text = "",
            confidence = 0.0f,
            isPartial = false,
            alternatives = emptyList()
        ))
        return@suspendCancellableCoroutine
    }
        
    try {
        val fileURL = NSURL.fileURLWithPath(audioFilePath)
        val request = SFSpeechURLRecognitionRequest(fileURL)
        
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = false
        
        println("Starting on-device file speech recognition")

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
        println("Stopping speech recognition...")
        audioEngine?.stop()
        audioEngine?.inputNode?.removeTapOnBus(0u)
        isRecording = false
    }
    
    recognitionRequest?.endAudio()
    recognitionRequest = null
    recognitionTask?.cancel()
    recognitionTask = null
    hasDetectedSpeech = false
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

fun isOnDeviceRecognitionAvailable(): Boolean {
    return speechRecognizer?.supportsOnDeviceRecognition() ?: false
} 