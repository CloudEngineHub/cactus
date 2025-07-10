#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

"$SCRIPT_DIR/build-ios.sh"
"$SCRIPT_DIR/build-react-android.sh"
"$SCRIPT_DIR/build-flutter-android.sh"

rm -rf "$ROOT_DIR/flutter/android/jniLibs.zip"
rm -rf "$ROOT_DIR/flutter/android/src/main/jniLibs/x86_64" 
rm -rf "$ROOT_DIR/react/android/src/main/jniLibs/x86_64"

"$SCRIPT_DIR/build-react.sh"
"$SCRIPT_DIR/build-flutter.sh"

cd "$ROOT_DIR/react"
npm version patch
npm publish

cd "$ROOT_DIR"
git add .
git commit -m "chore: publish"
git push origin main

cd "$ROOT_DIR/flutter"
flutter pub publish 