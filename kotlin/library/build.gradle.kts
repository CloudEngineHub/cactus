import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
}

group = "com.cactus"
version = "0.2.0"

kotlin {
    androidTarget {
        publishLibraryVariants("release")
        @OptIn(ExperimentalKotlinGradlePluginApi::class)
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }
    
    listOf(
        iosX64(),
        iosArm64(),
        iosSimulatorArm64()
    ).forEach { iosTarget ->
        iosTarget.binaries.framework {
            baseName = "CactusKotlin"
        }
        iosTarget.compilations.getByName("main") {
            cinterops {
                val cactus by creating {
                    defFile(project.file("src/iosMain/cinterop/cactus.def"))
                    packageName("com.cactus.native")
                    includeDirs(project.file("libs/ios/cactus.xcframework/ios-arm64/cactus.framework/Headers"))
                }
            }
        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                //put your multiplatform dependencies here
            }
        }
        val androidMain by getting {
            dependencies {
                implementation("net.java.dev.jna:jna:5.13.0")
            }
        }
    }
}

android {
    namespace = "com.cactus.library"
    compileSdk = libs.versions.android.compileSdk.get().toInt()
    defaultConfig {
        minSdk = libs.versions.android.minSdk.get().toInt()
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("libs/android/jniLibs")
        }
    }
}

mavenPublishing {
    pom {
        name = "Cactus"
        description = "Run AI locally in your apps"
        inceptionYear = "2025"
        url = "https://github.com/cactus-compute/cactus/"
    }
}
