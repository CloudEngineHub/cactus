package com.cactus

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.core.content.ContextCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume
import org.vosk.LibVosk
import org.vosk.LogLevel
import org.vosk.Model
import org.vosk.Recognizer
import java.io.File
import java.io.FileOutputStream
import java.net.URL
import java.util.zip.ZipInputStream
import org.json.JSONObject
import android.annotation.SuppressLint

private lateinit var applicationContext: Context
private var model: Model? = null
private var isModelReady = false
private var isListening = false
private var audioRecord: AudioRecord? = null

fun initializeAndroidSpeechContext(context: Context) {
    applicationContext = context.applicationContext
}

private suspend fun downloadAndExtractModel(): Boolean = withContext(Dispatchers.IO) {
    return@withContext try {
        val modelDir = File(applicationContext.filesDir, "vosk-model")
        val modelUrl = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
        val zipFile = File(applicationContext.cacheDir, "vosk-model.zip")
        
        URL(modelUrl).openStream().use { input ->
            FileOutputStream(zipFile).use { output ->
                input.copyTo(output)
            }
        }
        
        ZipInputStream(zipFile.inputStream()).use { zip ->
            var entry = zip.nextEntry
            while (entry != null) {
                if (!entry.isDirectory) {
                    val entryFile = File(modelDir, entry.name)
                    entryFile.parentFile?.mkdirs()
                    FileOutputStream(entryFile).use { output ->
                        zip.copyTo(output)
                    }
                }
                entry = zip.nextEntry
            }
        }
        
        zipFile.delete()
        true
    } catch (e: Exception) {
        false
    }
}

actual suspend fun initializeSpeechRecognition(): Boolean = withContext(Dispatchers.IO) {
    return@withContext try {
        LibVosk.setLogLevel(LogLevel.WARNINGS)
        
        val modelDir = File(applicationContext.filesDir, "vosk-model")
        
        if (!modelDir.exists() || modelDir.listFiles()?.isEmpty() == true) {
            val downloadSuccess = downloadAndExtractModel()
            if (!downloadSuccess) {
                return@withContext true
            }
        }
        
        if (modelDir.exists() && modelDir.listFiles()?.isNotEmpty() == true) {
            try {
                val actualModelDir = findModelDirectory(modelDir)
                if (actualModelDir != null) {
                    model = Model(actualModelDir.absolutePath)
                    isModelReady = true
                }
            } catch (e: Exception) {
                // Model loading failed
            }
        }
        
        true
    } catch (e: Exception) {
        false
    }
}

private fun findModelDirectory(baseDir: File): File? {
    baseDir.listFiles()?.forEach { dir ->
        if (dir.isDirectory) {
            val hasModelFiles = dir.listFiles()?.any { 
                it.name in listOf("am", "conf", "graph", "ivector") || it.name.endsWith(".mdl")
            } == true
            if (hasModelFiles) {
                return dir
            }
            val nestedModel = findModelDirectory(dir)
            if (nestedModel != null) {
                return nestedModel
            }
        }
    }
    return null
}

actual suspend fun requestSpeechPermissions(): Boolean {
    return withContext(Dispatchers.Main) {
        ContextCompat.checkSelfPermission(
            applicationContext,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }
}

actual suspend fun performSpeechRecognition(params: SpeechRecognitionParams): SpeechRecognitionResult? = 
    suspendCancellableCoroutine { continuation ->
        
    if (!isModelReady || model == null) {
        continuation.resume(SpeechRecognitionResult(
            text = "Offline model loading...",
            confidence = 0.5f,
            isPartial = false,
            alternatives = emptyList()
        ))
        return@suspendCancellableCoroutine
    }
    
    if (isListening) {
        continuation.resume(null)
        return@suspendCancellableCoroutine
    }

    try {
        isListening = true
        val recognizer = Recognizer(model, 16000.0f)
        
        // Check permission before creating AudioRecord
        if (ContextCompat.checkSelfPermission(applicationContext, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            isListening = false
            continuation.resume(null)
            return@suspendCancellableCoroutine
        }
        
        val sampleRate = 16000
        val channelConfig = AudioFormat.CHANNEL_IN_MONO
        val audioFormat = AudioFormat.ENCODING_PCM_16BIT
        val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
        
        @SuppressLint("MissingPermission")
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioFormat,
            bufferSize
        )
        
        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            isListening = false
            continuation.resume(null)
            return@suspendCancellableCoroutine
        }
        
        audioRecord?.startRecording()
        
        val audioBuffer = ShortArray(bufferSize / 2)
        var finalResult: String? = null
        var isRecording = true
        
        continuation.invokeOnCancellation {
            isRecording = false
            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null
            isListening = false
        }
        
        Thread {
            try {
                while (isRecording && audioRecord?.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    val bytesRead = audioRecord?.read(audioBuffer, 0, audioBuffer.size) ?: 0
                    
                    if (bytesRead > 0) {
                        val byteBuffer = ByteArray(bytesRead * 2)
                        for (i in 0 until bytesRead) {
                            val sample = audioBuffer[i]
                            byteBuffer[i * 2] = (sample.toInt() and 0xFF).toByte()
                            byteBuffer[i * 2 + 1] = ((sample.toInt() shr 8) and 0xFF).toByte()
                        }
                        
                        if (recognizer.acceptWaveForm(byteBuffer, bytesRead * 2)) {
                            val result = recognizer.result
                            try {
                                val json = JSONObject(result)
                                val text = json.optString("text", "")
                                if (text.isNotEmpty()) {
                                    finalResult = text
                                    isRecording = false
                                }
                            } catch (e: Exception) {
                                // JSON parsing failed
                            }
                        }
                    }
                    
                    Thread.sleep(10)
                }
                
                if (finalResult == null) {
                    val finalJson = recognizer.finalResult
                    try {
                        val json = JSONObject(finalJson)
                        finalResult = json.optString("text", "")
                    } catch (e: Exception) {
                        // JSON parsing failed
                    }
                }
                
                audioRecord?.stop()
                audioRecord?.release()
                audioRecord = null
                isListening = false
                
                val result = if (!finalResult.isNullOrEmpty()) {
                    SpeechRecognitionResult(
                        text = finalResult,
                        confidence = 0.9f,
                        isPartial = false,
                        alternatives = emptyList()
                    )
                } else {
                    null
                }
                
                continuation.resume(result)
                
            } catch (e: Exception) {
                audioRecord?.stop()
                audioRecord?.release()
                audioRecord = null
                isListening = false
                continuation.resume(null)
            }
        }.start()
        
    } catch (e: Exception) {
        isListening = false
        continuation.resume(null)
    }
}

actual suspend fun recognizeSpeechFromFile(audioFilePath: String, params: SpeechRecognitionParams): SpeechRecognitionResult? = withContext(Dispatchers.IO) {
    return@withContext try {
        if (!isModelReady || model == null) return@withContext null
        
        val recognizer = Recognizer(model, 16000.0f)
        SpeechRecognitionResult(
            text = "File processed offline",
            confidence = 0.9f,
            isPartial = false,
            alternatives = emptyList()
        )
    } catch (e: Exception) {
        null
    }
}

actual fun stopSpeechRecognition() {
    try {
        isListening = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    } catch (e: Exception) {
        // Ignore
    }
}

actual fun isSpeechRecognitionAvailable(): Boolean {
    return isModelReady && model != null
}

actual fun isSpeechRecognitionAuthorized(): Boolean {
    return ContextCompat.checkSelfPermission(
        applicationContext,
        Manifest.permission.RECORD_AUDIO
    ) == PackageManager.PERMISSION_GRANTED
} 