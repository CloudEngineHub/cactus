# copy ios/cactus.xcframework to and flutter/android/src/main/jniLibs to kotlin/library/src/commonMain/resources

# ./gradlew :library:clean :library:build :library:publishToMavenLocal
./gradlew :library:assembleCactusReleaseXCFramework
./gradlew :library:publishToMavenLocal
# ls -l ~/.m2/repository/com/cactus/
# ./gradlew :example:composeApp:assembleDebug
# From kotlin/ directory