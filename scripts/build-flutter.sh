SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR/cactus-flutter"

echo "Copying iOS frameworks to Flutter project..."
rm -rf ios/cactus.xcframework
cp -R "$ROOT_DIR/cactus-ios"/cactus.xcframework ios/
cp -R "$ROOT_DIR/cactus-ios"/CMakeLists.txt ios/