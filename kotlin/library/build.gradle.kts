import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import java.security.MessageDigest

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

// Enhanced task that automatically configures Xcode project
tasks.register("setupIOSFrameworkComplete") {
    description = "Copy the cactus.xcframework and automatically configure Xcode project"
    group = "cactus"
    
    dependsOn("setupIOSFramework")
    
    doLast {
        val projectFile = file("../example/iosApp/iosApp.xcodeproj/project.pbxproj")
        
        if (!projectFile.exists()) {
            throw GradleException("Xcode project not found at: ${projectFile.absolutePath}")
        }
        
        // Create backup
        val backupFile = file("${projectFile.absolutePath}.backup")
        projectFile.copyTo(backupFile, overwrite = true)
        
        try {
            val content = projectFile.readText()
            val modifiedContent = addFrameworkToXcodeProject(content, "cactus")
            projectFile.writeText(modifiedContent)
            
            println("✅ Framework copied and automatically configured in Xcode project")
            println("✅ Framework set to 'Embed & Sign'")
            println("🎉 Ready to build! Just open Xcode and build your project.")
        } catch (e: Exception) {
            // Restore backup on failure
            backupFile.copyTo(projectFile, overwrite = true)
            throw GradleException("Failed to modify Xcode project: ${e.message}. Backup restored.", e)
        }
    }
}

fun addFrameworkToXcodeProject(content: String, frameworkName: String): String {
    if (content.contains("$frameworkName.xcframework")) {
        println("Framework already exists in project, skipping modification")
        return content
    }
    
    // Generate unique UUIDs for Xcode project entries
    val fileRefUUID = generateXcodeUUID("${frameworkName}_fileRef")
    val buildFileUUID = generateXcodeUUID("${frameworkName}_buildFile") 
    val embedFileUUID = generateXcodeUUID("${frameworkName}_embedFile")
    val copyPhaseUUID = generateXcodeUUID("${frameworkName}_copyPhase")
    
    var modifiedContent = content
    
    // 1. Add to PBXFileReference section (add just before /* End PBXFileReference section */)
    val fileRefPattern = Regex("(/\\* End PBXFileReference section \\*/)")
    val fileRefEntry = "\t\t$fileRefUUID /* $frameworkName.xcframework */ = {isa = PBXFileReference; lastKnownFileType = wrapper.xcframework; path = iosApp/Frameworks/$frameworkName.xcframework; sourceTree = \"<group>\"; };\n"
    modifiedContent = modifiedContent.replace(fileRefPattern) {
        fileRefEntry + it.value
    }
    
    // 2. Add to PBXBuildFile section (add just before /* End PBXBuildFile section */)
    val buildFilePattern = Regex("(/\\* End PBXBuildFile section \\*/)")
    val buildFileEntry = "\t\t$buildFileUUID /* $frameworkName.xcframework in Frameworks */ = {isa = PBXBuildFile; fileRef = $fileRefUUID /* $frameworkName.xcframework */; };\n"
    val embedFileEntry = "\t\t$embedFileUUID /* $frameworkName.xcframework in Embed Frameworks */ = {isa = PBXBuildFile; fileRef = $fileRefUUID /* $frameworkName.xcframework */; settings = {ATTRIBUTES = (CodeSignOnCopy, RemoveHeadersOnCopy, ); }; };\n"
    modifiedContent = modifiedContent.replace(buildFilePattern) {
        buildFileEntry + embedFileEntry + it.value
    }
    
    // 3. Add to PBXFrameworksBuildPhase section - find the frameworks build phase
    val frameworkPhasePattern = Regex("(isa = PBXFrameworksBuildPhase;[\\s\\S]*?files = \\()([\\s\\S]*?)(\\);)")
    modifiedContent = modifiedContent.replace(frameworkPhasePattern) { match ->
        val prefix = match.groupValues[1]
        val existing = match.groupValues[2]
        val suffix = match.groupValues[3]
        val newEntry = if (existing.trim().isEmpty()) {
            "\n\t\t\t\t$buildFileUUID /* $frameworkName.xcframework in Frameworks */,\n\t\t\t"
        } else {
            "$existing\t\t\t\t$buildFileUUID /* $frameworkName.xcframework in Frameworks */,\n\t\t\t"
        }
        "$prefix$newEntry$suffix"
    }
    
    // 4. Add PBXCopyFilesBuildPhase section if not exists
    if (!modifiedContent.contains("Embed Frameworks")) {
        val copyPhasePattern = Regex("(/\\* End PBXCopyFilesBuildPhase section \\*/)")
        val copyPhaseEntry = "\t\t$copyPhaseUUID /* Embed Frameworks */ = {\n\t\t\tisa = PBXCopyFilesBuildPhase;\n\t\t\tbuildActionMask = 2147483647;\n\t\t\tdstPath = \"\";\n\t\t\tdstSubfolderSpec = 10;\n\t\t\tfiles = (\n\t\t\t\t$embedFileUUID /* $frameworkName.xcframework in Embed Frameworks */,\n\t\t\t);\n\t\t\tname = \"Embed Frameworks\";\n\t\t\trunOnlyForDeploymentPostprocessing = 0;\n\t\t};\n"
        modifiedContent = modifiedContent.replace(copyPhasePattern) {
            copyPhaseEntry + it.value
        }
        
        // Add to target buildPhases
        val buildPhasesPattern = Regex("(buildPhases = \\()([\\s\\S]*?)(\\);)")
        modifiedContent = modifiedContent.replace(buildPhasesPattern) { match ->
            val prefix = match.groupValues[1]
            val existing = match.groupValues[2]
            val suffix = match.groupValues[3]
            val newEntry = "$existing\t\t\t\t$copyPhaseUUID /* Embed Frameworks */,\n\t\t\t"
            "$prefix$newEntry$suffix"
        }
    }
    
    // 5. Add to Frameworks group (avoid duplicate addition)
    val frameworksGroupPattern = Regex("(42799AB246E5F90AF97AA0EF /\\* Frameworks \\*/ = \\{[\\s\\S]*?children = \\()([\\s\\S]*?)(\\);)")
    if (modifiedContent.contains("42799AB246E5F90AF97AA0EF")) {
        modifiedContent = modifiedContent.replace(frameworksGroupPattern) { match ->
            val prefix = match.groupValues[1]
            val existing = match.groupValues[2]
            val suffix = match.groupValues[3]
            val newEntry = if (existing.trim().isEmpty()) {
                "\n\t\t\t\t$fileRefUUID /* $frameworkName.xcframework */,\n\t\t\t"
            } else {
                "$existing\t\t\t\t$fileRefUUID /* $frameworkName.xcframework */,\n\t\t\t"
            }
            "$prefix$newEntry$suffix"
        }
    }
    
    return modifiedContent
}

fun generateXcodeUUID(seed: String): String {
    val digest = MessageDigest.getInstance("SHA-256")
    val hash = digest.digest(seed.toByteArray())
    return hash.take(12).joinToString("") { "%02X".format(it) }
}

// Make sure iOS framework is packaged when building
tasks.named("build") {
    dependsOn("packageIOSFramework")
}
