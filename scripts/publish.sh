#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

"$SCRIPT_DIR/build-ios.sh"
"$SCRIPT_DIR/build-react-android.sh"
rm -rf "$ROOT_DIR/react/android/src/main/jniLibs/x86_64"

"$SCRIPT_DIR/build-react.sh"
cd "$ROOT_DIR/react"
npm version patch
npm publish

cd "$ROOT_DIR"
git add .
git commit -m "publish latest xcframework and jnilibs"
git push origin main

# "$SCRIPT_DIR/build-flutter-android.sh"