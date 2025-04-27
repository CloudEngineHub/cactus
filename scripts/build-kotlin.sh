#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Building Kotlin package..."
cd "$ROOT_DIR/kotlin"

# Ensure the src directories exist
mkdir -p src/main/kotlin/com/cactus/kotlin/models
mkdir -p src/main/kotlin/com/cactus/kotlin/listeners
mkdir -p src/main/java/com/cactus/kotlin

# Ensure the AndroidManifest.xml exists
if [ ! -f "src/main/AndroidManifest.xml" ]; then
  cat > src/main/AndroidManifest.xml << 'EOF'
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <!-- The namespace is specified in build.gradle, no package attribute needed here -->
</manifest>
EOF
fi

echo "Building the Kotlin AAR package..."

# Clean and build the release AAR
./gradlew clean build

# Check if the build was successful
if [ -f "build/outputs/aar/cactus-kotlin-release.aar" ]; then
  echo "Kotlin build successful!"
  echo "Release AAR is located at: build/outputs/aar/cactus-kotlin-release.aar"
  echo "Debug AAR is located at: build/outputs/aar/cactus-kotlin-debug.aar"
else
  echo "Kotlin build failed!"
  exit 1
fi

# Optional: Copy the AAR files to a more accessible location
mkdir -p "$ROOT_DIR/dist"
cp build/outputs/aar/cactus-kotlin-release.aar "$ROOT_DIR/dist/"
cp build/outputs/aar/cactus-kotlin-debug.aar "$ROOT_DIR/dist/"

echo "Kotlin AAR files copied to $ROOT_DIR/dist/"
echo "Kotlin package built successfully!"