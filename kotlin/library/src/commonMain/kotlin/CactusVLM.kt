package com.cactus

class CactusVLM(
    private val threads: Int = 4,
    private val contextSize: Int = 2048,
    private val batchSize: Int = 512
) {
    private var handle: Long? = null
    
    suspend fun download(
        modelUrl: String = "https://huggingface.co/Cactus-Compute/SmolVLM2-500m-Instruct-GGUF/resolve/main/SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
        mmprojUrl: String = "https://huggingface.co/Cactus-Compute/SmolVLM2-500m-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
    ): Boolean {
        val modelFilename = modelUrl.substringAfterLast("/")
        val mmprojFilename = mmprojUrl.substringAfterLast("/")
        
        val modelSuccess = downloadVLMModel(modelUrl, modelFilename)
        val mmprojSuccess = downloadVLMModel(mmprojUrl, mmprojFilename)
        
        return modelSuccess && mmprojSuccess
    }
    
    suspend fun load(
        modelPath: String, 
        mmprojPath: String? = null
    ): Boolean {
        val actualMmprojPath = mmprojPath ?: modelPath.replace(".gguf", "-mmproj.gguf")
        handle = loadVLMModel(modelPath, actualMmprojPath, threads, contextSize, batchSize)
        return handle != null
    }
    
    suspend fun completion(
        prompt: String,
        imagePath: String,
        maxTokens: Int = 512,
        temperature: Float = 0.7f,
        topP: Float = 0.9f
    ): String? {
        return handle?.let { h ->
            generateVLMCompletion(h, prompt, imagePath, maxTokens, temperature, topP)
        }
    }
    
    fun unload() {
        handle?.let { h ->
            unloadVLMModel(h)
            handle = null
        }
    }
    
    fun isLoaded(): Boolean = handle != null
}

expect suspend fun downloadVLMModel(url: String, filename: String): Boolean
expect suspend fun loadVLMModel(modelPath: String, mmprojPath: String, threads: Int, contextSize: Int, batchSize: Int): Long?
expect suspend fun generateVLMCompletion(handle: Long, prompt: String, imagePath: String, maxTokens: Int, temperature: Float, topP: Float): String?
expect fun unloadVLMModel(handle: Long) 