package com.cactus.example

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.TimeUnit

private lateinit var applicationContext: Context

fun initializeAndroidContext(context: Context) {
    applicationContext = context.applicationContext
}



actual fun getModelCacheDir(): String {
    return applicationContext.cacheDir.absolutePath
}

actual suspend fun downloadModelStreaming(url: String, fileName: String, onProgress: (Float) -> Unit): String? {
    return withContext(Dispatchers.IO) {
        try {
            println("=== Starting model download ===")
            println("URL: $url")
            println("File: $fileName")
            
            val cacheDir = File(applicationContext.cacheDir, "models")
            if (!cacheDir.exists()) {
                val created = cacheDir.mkdirs()
                println("Created cache directory: $created")
            }
            
            val modelFile = File(cacheDir, fileName)
            println("Target file path: ${modelFile.absolutePath}")
            
            // If file already exists, return it
            if (modelFile.exists()) {
                println("File already exists, size: ${modelFile.length()} bytes")
                return@withContext modelFile.absolutePath
            }
            
            // Create OkHttp client with longer timeouts for large files
            val client = OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(300, TimeUnit.SECONDS)  // 5 minutes for large files
                .writeTimeout(30, TimeUnit.SECONDS)
                .followRedirects(true)
                .build()
            
            val request = Request.Builder()
                .url(url)
                .build()
            
            println("Making HTTP request...")
            val response = client.newCall(request).execute()
            
            if (!response.isSuccessful) {
                println("HTTP request failed: ${response.code} ${response.message}")
                response.close()
                return@withContext null
            }
            
            val contentLength = response.body?.contentLength() ?: -1L
            println("Content-Length: $contentLength bytes")
            
            val inputStream = response.body?.byteStream()
            if (inputStream == null) {
                println("Failed to get input stream")
                response.close()
                return@withContext null
            }
            
            val outputStream = FileOutputStream(modelFile)
            
            try {
                val buffer = ByteArray(8192) // 8KB buffer
                var totalBytesRead = 0L
                var bytesRead: Int
                
                println("Starting download loop...")
                while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                    outputStream.write(buffer, 0, bytesRead)
                    totalBytesRead += bytesRead
                    
                    // Update progress
                    if (contentLength > 0) {
                        val progress = totalBytesRead.toFloat() / contentLength.toFloat()
                        onProgress(progress)
                        
                        // Log progress every 10MB
                        if (totalBytesRead % (10 * 1024 * 1024) == 0L) {
                            println("Downloaded: ${totalBytesRead / (1024 * 1024)} MB / ${contentLength / (1024 * 1024)} MB")
                        }
                    }
                }
                
                outputStream.flush()
                println("Download completed. Total bytes: $totalBytesRead")
                
                // Verify file
                if (modelFile.exists() && modelFile.length() > 0) {
                    println("✓ File verification successful. Size: ${modelFile.length()} bytes")
                    modelFile.absolutePath
                } else {
                    println("✗ File verification failed")
                    if (modelFile.exists()) {
                        modelFile.delete()
                    }
                    null
                }
                
            } finally {
                try {
                    outputStream.close()
                    inputStream.close()
                    response.close()
                } catch (e: Exception) {
                    println("Error closing streams: ${e.message}")
                }
            }
            
        } catch (e: IOException) {
            println("IOException during download: ${e.message}")
            e.printStackTrace()
            null
        } catch (e: Exception) {
            println("Unexpected error during download: ${e.message}")
            e.printStackTrace()
            null
        }
    }
} 