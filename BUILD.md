# LocalImage Android build

The CI workflow installs Gradle 9.4.1 explicitly and builds with the configured Android toolchain.

AGP 9.x provides Kotlin support directly, so `org.jetbrains.kotlin.android` is intentionally not applied in `app/build.gradle.kts`.
The Compose compiler plugin remains enabled.

CI commands:

```text
gradle --no-daemon --stacktrace :app:assembleDebug
gradle --no-daemon --stacktrace :app:assembleRelease
```

Artifacts are collected under `dist/` and published to both the Actions artifact and the continuous GitHub Release.
