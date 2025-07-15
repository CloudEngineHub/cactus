#!/bin/bash

# Cactus iOS Framework Setup Script
# This script helps integrate the cactus.xcframework into your iOS project

set -e

echo "🌵 Cactus iOS Framework Setup"
echo "================================"

# Check if we're in the right directory
if [ ! -f "build.gradle.kts" ]; then
    echo "❌ Error: This script must be run from your Kotlin project root directory"
    exit 1
fi

# Function to find the framework in dependencies
find_framework() {
    local gradle_cache="$HOME/.gradle/caches/modules-2/files-2.1/com.cactus/cactus-kmp"
    if [ -d "$gradle_cache" ]; then
        find "$gradle_cache" -name "cactus.xcframework" -type d 2>/dev/null | head -1
    fi
}

# Function to extract framework from JAR
extract_framework() {
    local jar_path="$1"
    local extract_dir="$2"
    
    echo "📦 Extracting framework from library..."
    unzip -q "$jar_path" -d "$extract_dir" || true
    
    # Find the extracted framework
    find "$extract_dir" -name "cactus.xcframework" -type d 2>/dev/null | head -1
}

# Try to find the framework
framework_path=$(find_framework)

if [ -z "$framework_path" ]; then
    echo "⚠️  Framework not found in Gradle cache. Checking library JAR..."
    
    # Look for the library JAR
    jar_path=$(find "$HOME/.gradle/caches/modules-2/files-2.1/com.cactus/cactus-kmp" -name "*.jar" 2>/dev/null | head -1)
    
    if [ -n "$jar_path" ]; then
        temp_dir=$(mktemp -d)
        framework_path=$(extract_framework "$jar_path" "$temp_dir")
    fi
fi

if [ -z "$framework_path" ]; then
    echo "❌ Error: Could not find cactus.xcframework"
    echo "   Make sure you have added the dependency to your build.gradle.kts:"
    echo "   implementation(\"com.cactus:cactus-kmp:0.2.0\")"
    echo "   And run './gradlew build' first"
    exit 1
fi

echo "✅ Found framework at: $framework_path"

# Check if iOS project exists
ios_project=$(find . -name "*.xcodeproj" | head -1)
if [ -z "$ios_project" ]; then
    echo "❌ Error: No Xcode project found"
    echo "   This script should be run from your Kotlin Multiplatform project root"
    exit 1
fi

# Copy framework to iOS project
ios_dir=$(dirname "$ios_project")
target_path="$ios_dir/Frameworks"

echo "📁 Creating Frameworks directory..."
mkdir -p "$target_path"

echo "📋 Copying framework..."
cp -R "$framework_path" "$target_path/"

echo "✅ Framework copied to: $target_path/cactus.xcframework"

echo ""
echo "🎯 Next Steps:"
echo "1. Open your Xcode project"
echo "2. Right-click on your project root and select 'Add Files to [ProjectName]'"
echo "3. Navigate to: $target_path/cactus.xcframework"
echo "4. Select the framework and click 'Add'"
echo "5. In your app target settings, go to 'Frameworks, Libraries, and Embedded Content'"
echo "6. Find 'cactus.xcframework' and set it to 'Embed & Sign'"
echo ""
echo "🚀 Your iOS project is now ready to use the Cactus library!" 