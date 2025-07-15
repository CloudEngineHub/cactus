package com.cactus.example

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.runtime.LaunchedEffect

class MainActivity : ComponentActivity() {
    
    companion object {
        init {
            try {
                // Configure JNA for Android
                System.setProperty("jna.nosys", "true")
                System.setProperty("jna.noclasspath", "true")
                
                // Load the native library
                System.loadLibrary("cactus")
                println("Native library loaded successfully")
            } catch (e: UnsatisfiedLinkError) {
                println("Failed to load native library: ${e.message}")
            }
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize Android context for model file handling
        initializeAndroidContext(this)
        
        enableEdgeToEdge()
        setContent {
            LaunchedEffect(isSystemInDarkTheme()) {
                enableEdgeToEdge()
            }
            App()
        }
    }
} 