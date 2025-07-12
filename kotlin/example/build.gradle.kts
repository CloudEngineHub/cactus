plugins {
    kotlin("jvm")
    application
}

repositories {
    mavenLocal()
    mavenCentral()
}

dependencies {
    implementation(project(":library"))
}

application {
    mainClass.set("com.cactus.example.MainKt")
} 