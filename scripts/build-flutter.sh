SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Building Android JNILibs..."
"$SCRIPT_DIR/build-react-android.sh"

cd "$ROOT_DIR/cactus-flutter"

echo "Copying iOS frameworks to Flutter project..."
rm -rf ios/cactus.xcframework
cp -R "$ROOT_DIR/cactus-ios"/cactus.xcframework ios/
cp -R "$ROOT_DIR/cactus-ios"/CMakeLists.txt ios/

echo "Building Cactus Flutter Plugin..."
flutter clean
flutter pub get
flutter build

echo "Build completed successfully."