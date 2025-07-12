package com.cactus.example

import com.cactus.generateFibi

fun main() {
    println("Testing Cactus Library")
    println("====================")
    
    val sequence = generateFibi()
    val first10 = sequence.take(10).toList()
    
    println("First 10 numbers in the sequence:")
    first10.forEachIndexed { index, value ->
        println("${index + 1}: $value")
    }
    
    println("\nLibrary test completed successfully!")
} 