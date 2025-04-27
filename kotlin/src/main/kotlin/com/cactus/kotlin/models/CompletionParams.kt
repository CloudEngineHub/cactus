package com.cactus.kotlin.models

/**
 * Parameters for text completion
 */
data class CompletionParams(
    /** The prompt to generate from */
    val prompt: String,
    
    /** Maximum tokens to generate */
    val maxTokens: Int = 128,
    
    /** Temperature (randomness) */
    val temperature: Float = 0.8f,
    
    /** Top-p sampling */
    val topP: Float = 0.9f,
    
    /** Top-k sampling */
    val topK: Int = 40,
    
    /** Frequency penalty */
    val frequencyPenalty: Float = 0.0f,
    
    /** Presence penalty */
    val presencePenalty: Float = 0.0f,
    
    /** Repetition penalty */
    val repetitionPenalty: Float = 1.1f,
    
    /** Mirostat sampling */
    val mirostat: Int = 0,
    
    /** Mirostat tau */
    val mirostatTau: Float = 5.0f,
    
    /** Mirostat eta */
    val mirostatEta: Float = 0.1f,
    
    /** Grammar for constrained sampling */
    val grammar: String? = null,
    
    /** JSON schema for constraining JSON output */
    val jsonSchema: String? = null,
    
    /** End sequences to terminate generation */
    val stopSequences: List<String>? = null,
    
    /** Seed for random generation */
    val seed: Int = -1,
    
    /** Logit bias */
    val logitBias: Map<Int, Float>? = null,
    
    /** System prompt */
    val system: String? = null,
    
    /** Chat template name */
    val template: String? = null,
    
    /** Enable TensorRT-LLM engine */
    val trtEnabled: Boolean = false
) 