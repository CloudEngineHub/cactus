# cactus_example

A new Flutter project.

## Getting Started

First, clone the repo with `git clone https://github.com/cactus-compute/cactus.git`, cd into it and make all scripts executable with `chmod +x scripts/*.sh`

- Build the Android JNILibs by running `scripts/build-flutter-android.sh`.
- Build the Flutter Plugin  by running `scripts/build-flutter.sh`. (MUST run before using example)
- Navigate to the example app with `cd flutter/example`.
- Open your simulator via Xcode or Android Studio, [walkthrough](https://medium.com/@daspinola/setting-up-android-and-ios-emulators-22d82494deda) if you have not done this before.
- Always start app with this combo `flutter clean && flutter pub get && flutter run`.
- Play with the app, and make changes either to the example app or plugin as desired.

## Resources

A few resources to get you started if this is your first Flutter project:

- [Lab: Write your first Flutter app](https://docs.flutter.dev/get-started/codelab)
- [Cookbook: Useful Flutter samples](https://docs.flutter.dev/cookbook)

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev/), which offers tutorials,
samples, guidance on mobile development, and a full API reference.
