package com.cactus.example

import android.content.Context
import io.ktor.client.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.utils.io.*
import kotlinx.coroutines.yield
import java.io.File
import java.io.FileOutputStream

private lateinit var applicationContext: Context

fun initializeAndroidContext(context: Context) {
    applicationContext = context.applicationContext
}

actual fun getModelCacheDir(): String {
    return applicationContext.cacheDir.absolutePath
}

actual suspend fun downloadModelStreaming(url: String, fileName: String, onProgress: (Float) -> Unit): String? {
    return try {
        val httpClient = HttpClient()
        
        try {
            val cacheDir = File(applicationContext.cacheDir, "models")
            if (!cacheDir.exists()) {
                cacheDir.mkdirs()
            }
            
            val modelFile = File(cacheDir, fileName)
            
            // If file already exists, return it
            if (modelFile.exists()) {
                return modelFile.absolutePath
            }
            
            val response = httpClient.get(url)
            val contentLength = response.headers["Content-Length"]?.toLongOrNull() ?: 0L
            
            val channel = response.bodyAsChannel()
            val outputStream = FileOutputStream(modelFile)
            
            var downloadedBytes = 0L
            val buffer = ByteArray(8192) // 8KB buffer
            
            try {
                while (!channel.isClosedForRead) {
                    val bytesRead = channel.readAvailable(buffer, 0, buffer.size)
                    if (bytesRead == -1) break
                    
                    outputStream.write(buffer, 0, bytesRead)
                    downloadedBytes += bytesRead
                    
                    if (contentLength > 0) {
                        val progress = downloadedBytes.toFloat() / contentLength.toFloat()
                        onProgress(progress)
                    }
                    
                    yield() // Allow other coroutines to run
                }
            } finally {
                outputStream.close()
            }
            
            modelFile.absolutePath
        } finally {
            httpClient.close()
        }
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
} 