import org.jlleitschuh.gradle.ktlint.reporter.ReporterType

plugins {
    kotlin("jvm") version "2.3.0"
    application
    id("org.jlleitschuh.gradle.ktlint") version "14.0.1"
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

ktlint {
    version.set("1.8.0")
    verbose.set(true)
    outputToConsole.set(true)
    coloredOutput.set(true)
    reporters {
        reporter(ReporterType.CHECKSTYLE)
        reporter(ReporterType.JSON)
        reporter(ReporterType.HTML)
    }
    filter {
        exclude("**/style-violations.kt")
    }
}
