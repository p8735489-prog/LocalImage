#!/bin/sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"
test -f app/build.gradle.kts
test -f app/src/main/AndroidManifest.xml
test -f app/src/main/cpp/CMakeLists.txt
test -f .github/workflows/android-apk.yml
test -x gradlew
if [ -f app/build/outputs/apk/debug/app-debug.apk ]; then
  echo "DEBUG_APK=present"
else
  echo "DEBUG_APK=not-built"
fi
if [ -f app/build/outputs/apk/release/app-release-unsigned.apk ]; then
  echo "RELEASE_UNSIGNED_APK=present"
else
  echo "RELEASE_UNSIGNED_APK=not-built"
fi
