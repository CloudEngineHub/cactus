# copy JNI libs to kotlin/library/libs/android
# copy ios/cactus.xcframework to kotlin/library/libs/ios

# ./gradlew clean build
# ./gradlew :library:assembleRelease --quiet && echo "Android library built successfully" && ls -la library/build/outputs/aar/
# ./gradlew :library:iosX64MainKlibrary --quiet && echo "iOS library built successfully" && ls -la library/build/classes/kotlin/iosX64/main/
# ./gradlew publishToMavenLocal
# ls -la ~/.m2/repository/com/cactus/cactus/0.2.0/
# ./gradlew :example:run
