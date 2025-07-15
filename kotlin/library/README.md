# Cactus Kotlin Multiplatform Library

A Kotlin Multiplatform library for running AI models locally on Android and iOS.

## Installation

### For Android

Add the dependency to your `build.gradle.kts`:

```kotlin
dependencies {
    implementation("com.cactus:cactus-kmp:0.2.0")
}
```

### For iOS

#### Option 1: Gradle Task (Recommended)

1. Add the dependency to your `build.gradle.kts`:

```kotlin
dependencies {
    implementation("com.cactus:cactus-kmp:0.2.0")
}
```

2. Run the setup task:

```bash
./gradlew :library:setupIOSFramework
```

3. In Xcode:
   - Right-click your project → "Add Files to [ProjectName]"
   - Navigate to `iosApp/iosApp/Frameworks/cactus.xcframework`
   - Select and click "Add"
   - In target settings → "Frameworks, Libraries, and Embedded Content"
   - Set `cactus.xcframework` to "Embed & Sign"

#### Option 2: Manual Setup

1. Add the dependency to your `build.gradle.kts`
2. Run the setup script:

```bash
bash library/scripts/setup_ios_framework.sh
```

3. Follow the printed instructions to add the framework to your Xcode project

#### Option 3: Complete Manual Integration

1. Add the dependency and build your project
2. Find the framework in your Gradle cache:
   ```bash
   find ~/.gradle/caches -name "cactus.xcframework" -type d
   ```
3. Copy the framework to your iOS project directory
4. Add to Xcode as described above

## Usage

```kotlin
// Common code
import com.cactus.library.CactusContext

val context = CactusContext()
val result = context.downloadFile("https://example.com/model.gguf", "/path/to/model.gguf") { progress ->
    println("Progress: $progress")
}
```

## Platform-Specific Features

### Android
- Uses OkHttp for reliable downloads
- JNA integration for native library access
- Automatic native library loading

### iOS
- Uses NSURLSession for native downloads
- XCFramework with multi-architecture support
- Automatic framework embedding via CocoaPods

## Requirements

- **Android**: API 24+
- **iOS**: iOS 13.0+
- **Kotlin**: 2.1.10+

## Native Dependencies

### Android
- `libcactus.so` (automatically included)
- JNA for native interop

### iOS
- `cactus.xcframework` (automatically embedded via CocoaPods)
- System frameworks: Accelerate, Foundation, Metal, MetalKit 