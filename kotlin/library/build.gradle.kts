import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidLibrary)
    `maven-publish`
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
        iosTarget.compilations.getByName("main") {
            cinterops {
                val cactus by creating {
                    defFile(project.file("src/iosMain/cinterop/cactus.def"))
                    packageName("com.cactus.native")
                    
                    // Determine the correct architecture path
                    val archPath = when (iosTarget.name) {
                        "iosArm64" -> "ios-arm64"
                        "iosX64" -> "ios-arm64_x86_64-simulator"  
                        "iosSimulatorArm64" -> "ios-arm64_x86_64-simulator"
                        else -> "ios-arm64"
                    }
                    
                    val frameworkPath = project.file("libs/ios/cactus.xcframework/$archPath")
                    includeDirs(project.file("$frameworkPath/cactus.framework/Headers"))
                    
                    // Set compiler options dynamically
                    compilerOpts("-framework", "Accelerate", "-framework", "Foundation", 
                               "-framework", "Metal", "-framework", "MetalKit", 
                               "-framework", "cactus", "-F", frameworkPath.absolutePath)
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

publishing {
    publications {
        withType<MavenPublication> {
            pom {
                name.set("Cactus")
                description.set("Run AI locally in your apps")
                url.set("https://github.com/cactus-compute/cactus/")
            }
        }
    }
}

// Package XCFramework as part of the library resources
tasks.register<Copy>("packageIOSFramework") {
    from("libs/ios/cactus.xcframework")
    into("$buildDir/resources/ios/cactus.xcframework")
}

// Task to help developers set up the iOS framework
tasks.register<Copy>("setupIOSFramework") {
    description = "Copy the cactus.xcframework to your iOS project"
    group = "cactus"
    
    from("libs/ios/cactus.xcframework")
    into("../example/iosApp/iosApp/Frameworks/cactus.xcframework")
    
    doLast {
        println("✅ Framework copied to iOS project")
        println("Next steps:")
        println("1. Open your Xcode project")
        println("2. Add cactus.xcframework to your project")
        println("3. Set it to 'Embed & Sign' in your target settings")
    }
}

// Make sure iOS framework is packaged when building
tasks.named("build") {
    dependsOn("packageIOSFramework")
}
