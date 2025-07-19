package com.cactus

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.net.URL

private var currentHandle: Long? = null

private val applicationContext: Context by lazy {
    CactusContextInitializer.getApplicationContext()
}

actual suspend fun downloadModel(url: String, filename: String): Boolean {
    return withContext(Dispatchers.IO) {
        try {
            val modelsDir = File(applicationContext.cacheDir, "models")
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

actual suspend fun loadModel(path: String, threads: Int, contextSize: Int, batchSize: Int): Long? {
    return withContext(Dispatchers.Default) {
        try {
            val modelsDir = File(applicationContext.cacheDir, "models")
            val modelFile = File(modelsDir, path)
            
            if (!modelFile.exists()) return@withContext null
            
            val params = CactusInitParams(
                modelPath = modelFile.absolutePath,
                nCtx = contextSize,
                nThreads = threads,
                nBatch = batchSize
            )
            
            val handle = CactusContext.initContext(params)
            if (handle != null) {
                currentHandle = handle
                handle
            } else {
                null
            }
        } catch (e: Exception) {
            null
        }
    }
}

actual suspend fun generateCompletion(handle: Long, prompt: String, maxTokens: Int, temperature: Float, topP: Float): String? {
    return withContext(Dispatchers.Default) {
        try {
            val params = CactusCompletionParams(
                prompt = prompt,
                nPredict = maxTokens,
                temperature = temperature.toDouble(),
                topP = topP.toDouble()
            )
            
            val result = CactusContext.completion(handle, params)
            result.text
        } catch (e: Exception) {
            null
        }
    }
}

actual fun unloadModel(handle: Long) {
    try {
        CactusContext.freeContext(handle)
        if (currentHandle == handle) {
            currentHandle = null
        }
    } catch (e: Exception) {
    }
} 