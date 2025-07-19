@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
package com.cactus.example

import kotlinx.cinterop.*
import kotlinx.coroutines.suspendCancellableCoroutine
import platform.Foundation.*
import platform.darwin.NSObject
import kotlin.coroutines.resume

 