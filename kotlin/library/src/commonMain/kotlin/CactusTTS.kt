package com.cactus

data class TTSResult(
    val audioPath: String,
    val duration: Float,
    val success: Boolean
)

class CactusTTS(
    private val voice: String = "default",
    private val speed: Float = 1.0f,
    private val pitch: Float = 1.0f,
    private val sampleRate: Int = 22050
) {
    private var isInitialized = false
    
    suspend fun download(url: String): Boolean {
        val filename = url.substringAfterLast("/")
        return downloadTTSModel(url, filename)
    }
    
    suspend fun initialize(): Boolean {
        isInitialized = initializeTTS()
        return isInitialized
    }
    
    suspend fun synthesize(text: String, outputPath: String? = null): TTSResult? {
        return if (isInitialized) {
            performTTS(text, outputPath, voice, speed, pitch, sampleRate)
        } else null
    }
    
    suspend fun speak(text: String): Boolean {
        return if (isInitialized) {
            speakTTS(text, voice, speed, pitch)
        } else false
    }
    
    fun stop() {
        stopTTS()
    }
    
    fun isReady(): Boolean = isInitialized
}

expect suspend fun downloadTTSModel(url: String, filename: String): Boolean
expect suspend fun initializeTTS(): Boolean
expect suspend fun performTTS(text: String, outputPath: String?, voice: String, speed: Float, pitch: Float, sampleRate: Int): TTSResult?
expect suspend fun speakTTS(text: String, voice: String, speed: Float, pitch: Float): Boolean
expect fun stopTTS() 