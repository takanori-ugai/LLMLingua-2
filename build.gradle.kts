plugins {
    kotlin("jvm") version "2.3.0"
    application
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("ai.djl:api:0.36.0")
    implementation("ai.djl.onnxruntime:onnxruntime-engine:0.36.0")
    implementation("ai.djl.huggingface:tokenizers:0.36.0")
    testImplementation(kotlin("test"))
}

application {
    mainClass.set("org.example.MainKt")
}

kotlin {
    jvmToolchain(21)
}

tasks.test {
    useJUnitPlatform()
}