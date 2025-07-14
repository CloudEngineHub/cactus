package com.cactus.example

import io.ktor.client.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.utils.io.*
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned
import kotlinx.coroutines.yield
import platform.Foundation.*

actual fun getModelCacheDir(): String {
    val cacheDir = NSSearchPathForDirectoriesInDomains(
        NSCachesDirectory,
        NSUserDomainMask,
        true
    ).firstObject() as NSString
    return cacheDir.toString()
}

actual suspend fun downloadModelStreaming(url: String, fileName: String, onProgress: (Float) -> Unit): String? {
    return try {
        val httpClient = HttpClient()
        
        try {
            val cacheDir = getModelCacheDir()
            val modelsDir = "$cacheDir/models"
            
            // Create models directory if it doesn't exist
            val fileManager = NSFileManager.defaultManager
            fileManager.createDirectoryAtPath(
                modelsDir,
                withIntermediateDirectories = true,
                attributes = null,
                error = null
            )
            
            val filePath = "$modelsDir/$fileName"
            
            // If file already exists, return it
            if (fileManager.fileExistsAtPath(filePath)) {
                return filePath
            }
            
            val response = httpClient.get(url)
            val contentLength = response.headers["Content-Length"]?.toLongOrNull() ?: 0L
            
            val channel = response.bodyAsChannel()
            val outputStream = NSOutputStream.outputStreamToFileAtPath(filePath, append = false)
            outputStream?.open()
            
            var downloadedBytes = 0L
            val buffer = ByteArray(8192) // 8KB buffer
            
            try {
                while (!channel.isClosedForRead) {
                    val bytesRead = channel.readAvailable(buffer, 0, buffer.size)
                    if (bytesRead == -1) break
                    
                    // Write to NSOutputStream
                    buffer.usePinned { pinned ->
                        outputStream?.write(pinned.addressOf(0), bytesRead.toULong())
                    }
                    
                    downloadedBytes += bytesRead
                    
                    if (contentLength > 0) {
                        val progress = downloadedBytes.toFloat() / contentLength.toFloat()
                        onProgress(progress)
                    }
                    
                    yield() // Allow other coroutines to run
                }
            } finally {
                outputStream?.close()
            }
            
            filePath
        } finally {
            httpClient.close()
        }
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
} 