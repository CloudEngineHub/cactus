package com.cactus

import android.content.Context

object CactusContextInitializer {
    private var applicationContext: Context? = null
    
    init {
        try {
            System.setProperty("jna.nosys", "true")
            System.setProperty("jna.noclasspath", "true")
            System.loadLibrary("cactus")
        } catch (e: UnsatisfiedLinkError) {
            e.printStackTrace()
        }
    }
    
    fun initialize(context: Context) {
        if (applicationContext == null) {
            applicationContext = context.applicationContext
            initializeAndroidSpeechContext(context)
        }
    }
    
    fun getApplicationContext(): Context {
        return applicationContext ?: throw IllegalStateException(
            "CactusContextInitializer not initialized. Call CactusContextInitializer.initialize(context) in your Application or Activity."
        )
    }
} 