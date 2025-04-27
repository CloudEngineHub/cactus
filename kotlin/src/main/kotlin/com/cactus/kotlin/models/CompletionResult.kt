package com.cactus.kotlin.models

/**
 * Result of a text completion
 */
data class CompletionResult(
    /** Generated text */
    val text: String,
    
    /** Time taken for generation in milliseconds */
    val timeTaken: Long,
    
    /** Total tokens used */
    val totalTokens: Int,
    
    /** Tokens per second */
    val tokensPerSecond: Float,
    
    /** Tokens in prompt */
    val promptTokens: Int,
    
    /** Tokens in completion */
    val completionTokens: Int,
    
    /** Whether the generation was truncated */
    val truncated: Boolean = false,
    
    /** Reason for stopping */
    val stopReason: String? = null
) 