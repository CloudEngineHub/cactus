#!/bin/bash -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$ANDROID_HOME" ]; then
  if [ -d "$HOME/Library/Android/sdk" ]; then
    export ANDROID_HOME="$HOME/Library/Android/sdk"
  elif [ -d "$HOME/Android/Sdk" ]; then
    export ANDROID_HOME="$HOME/Android/Sdk"
  else
    echo "ANDROID_HOME not set and Android SDK not found in default locations"
    echo "Please set ANDROID_HOME environment variable"
    exit 1
  fi
  echo "ANDROID_HOME set to: $ANDROID_HOME"
fi

NDK_VERSION=27.0.12077973
CMAKE_TOOLCHAIN_FILE=$ANDROID_HOME/ndk/$NDK_VERSION/build/cmake/android.toolchain.cmake
ANDROID_PLATFORM=android-21
CMAKE_BUILD_TYPE=Release

FLUTTER_PLUGIN_ANDROID_MAIN_SRC_DIR="$ROOT_DIR/android/src/main"

if [ ! -d "$ANDROID_HOME/ndk/$NDK_VERSION" ]; then
  echo "NDK $NDK_VERSION not found, available versions: $(ls $ANDROID_HOME/ndk)"
  echo "Run \$ANDROID_HOME/tools/bin/sdkmanager \"ndk;$NDK_VERSION\""
  CMAKE_VERSION=3.10.2.4988404
  echo "and \$ANDROID_HOME/tools/bin/sdkmanager \"cmake;$CMAKE_VERSION\""
  exit 1
fi

if ! command -v cmake &> /dev/null; then
  echo "cmake could not be found, please install it"
  exit 1
fi

n_cpu=1
if uname -a | grep -q "Darwin"; then
  n_cpu=$(sysctl -n hw.logicalcpu)
elif uname -a | grep -q "Linux"; then
  n_cpu=$(nproc)
fi

t0=$(date +%s)

cd "$FLUTTER_PLUGIN_ANDROID_MAIN_SRC_DIR"

ABI_ARM64="arm64-v8a"
BUILD_DIR_ARM64="build-arm64"
JNI_DEST_DIR_ARM64="jniLibs/$ABI_ARM64"

echo "Building for $ABI_ARM64..."
cmake -DCMAKE_TOOLCHAIN_FILE="$CMAKE_TOOLCHAIN_FILE" \
  -DANDROID_ABI="$ABI_ARM64" \
  -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
  -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
  -S . \
  -B "$BUILD_DIR_ARM64"

cmake --build "$BUILD_DIR_ARM64" --config "$CMAKE_BUILD_TYPE" -j "$n_cpu"

echo "Copying $ABI_ARM64 libraries..."
mkdir -p "$JNI_DEST_DIR_ARM64"
cp "$BUILD_DIR_ARM64"/lib*.so "$JNI_DEST_DIR_ARM64/"

rm -rf "$BUILD_DIR_ARM64"
echo "$ABI_ARM64 build and copy complete."

ABI_X86_64="x86_64"
BUILD_DIR_X86_64="build-x86_64"
JNI_DEST_DIR_X86_64="jniLibs/$ABI_X86_64"

echo "Building for $ABI_X86_64..."
cmake -DCMAKE_TOOLCHAIN_FILE="$CMAKE_TOOLCHAIN_FILE" \
  -DANDROID_ABI="$ABI_X86_64" \
  -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
  -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
  -S . \
  -B "$BUILD_DIR_X86_64"

cmake --build "$BUILD_DIR_X86_64" --config "$CMAKE_BUILD_TYPE" -j "$n_cpu"

echo "Copying $ABI_X86_64 libraries..."
mkdir -p "$JNI_DEST_DIR_X86_64"
cp "$BUILD_DIR_X86_64"/lib*.so "$JNI_DEST_DIR_X86_64/"

rm -rf "$BUILD_DIR_X86_64"
echo "$ABI_X86_64 build and copy complete."


t1=$(date +%s)
echo "Total time: $((t1 - t0)) seconds"
echo "Native libraries successfully built and copied to $FLUTTER_PLUGIN_ANDROID_MAIN_SRC_DIR/jniLibs"
