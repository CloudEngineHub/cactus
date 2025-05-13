SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# echo "Building Android JNILibs..."
# "$SCRIPT_DIR/build-react-android.sh"

cd "$ROOT_DIR/cactus-flutter"

echo "Copying iOS frameworks to Flutter project..."
rm -rf ios/cactus.xcframework
cp -R "$ROOT_DIR/cactus-ios"/cactus.xcframework ios/
cp -R "$ROOT_DIR/cactus-ios"/CMakeLists.txt ios/

echo "Zipping JNILibs and XCFramework..."

# Define source and target paths
JNI_LIBS_SOURCE_PARENT_DIR="android/src/main"
JNI_LIBS_FOLDER_NAME="jniLibs"
JNI_LIBS_ZIP_TARGET="android/jniLibs.zip" # Target relative to cactus-flutter

XCFRAMEWORK_SOURCE_PARENT_DIR="ios"
XCFRAMEWORK_FOLDER_NAME="cactus.xcframework"
XCFRAMEWORK_ZIP_TARGET="ios/cactus.xcframework.zip" # Target relative to cactus-flutter

# Zip JNILibs
# Assumes artifacts are in $ROOT_DIR/cactus-flutter/android/src/main/jniLibs
if [ -d "$JNI_LIBS_SOURCE_PARENT_DIR/$JNI_LIBS_FOLDER_NAME" ]; then
  echo "Zipping JNILibs from $JNI_LIBS_SOURCE_PARENT_DIR/$JNI_LIBS_FOLDER_NAME..."
  # cd into parent of jniLibs, zip to ../../jniLibs.zip (which becomes android/jniLibs.zip)
  (cd "$JNI_LIBS_SOURCE_PARENT_DIR" && zip -r "../../jniLibs.zip" "$JNI_LIBS_FOLDER_NAME")
  if [ $? -eq 0 ]; then
    echo "JNILibs successfully zipped to $JNI_LIBS_ZIP_TARGET"
  else
    echo "Error: Failed to zip JNILibs."
  fi
elif [ -f "$JNI_LIBS_ZIP_TARGET" ]; then
  echo "Warning: JNILibs source directory $JNI_LIBS_SOURCE_PARENT_DIR/$JNI_LIBS_FOLDER_NAME not found, but $JNI_LIBS_ZIP_TARGET already exists. Skipping zip."
else
  echo "Error: JNILibs source directory $JNI_LIBS_SOURCE_PARENT_DIR/$JNI_LIBS_FOLDER_NAME not found. Cannot zip."
fi

# Zip XCFramework
if [ -d "$XCFRAMEWORK_SOURCE_PARENT_DIR/$XCFRAMEWORK_FOLDER_NAME" ]; then
  echo "Zipping XCFramework from $XCFRAMEWORK_SOURCE_PARENT_DIR/$XCFRAMEWORK_FOLDER_NAME..."
  # cd into parent of cactus.xcframework (ios), zip to cactus.xcframework.zip (which becomes ios/cactus.xcframework.zip)
  (cd "$XCFRAMEWORK_SOURCE_PARENT_DIR" && zip -r "cactus.xcframework.zip" "$XCFRAMEWORK_FOLDER_NAME")
  if [ $? -eq 0 ]; then
    echo "XCFramework successfully zipped to $XCFRAMEWORK_ZIP_TARGET"
  else
    echo "Error: Failed to zip XCFramework."
  fi
elif [ -f "$XCFRAMEWORK_ZIP_TARGET" ]; then
  echo "Warning: XCFramework source directory $XCFRAMEWORK_SOURCE_PARENT_DIR/$XCFRAMEWORK_FOLDER_NAME not found, but $XCFRAMEWORK_ZIP_TARGET already exists. Skipping zip."
else
  echo "Error: XCFramework source directory $XCFRAMEWORK_SOURCE_PARENT_DIR/$XCFRAMEWORK_FOLDER_NAME not found. Cannot zip."
fi

echo "Building Cactus Flutter Plugin..."
flutter clean
flutter pub get
dart analyze

echo "Build completed successfully."