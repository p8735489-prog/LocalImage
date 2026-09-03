#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
required=(
  README.md
  gradlew
  gradlew.bat
  gradle/wrapper/gradle-wrapper.properties
  settings.gradle.kts
  build.gradle.kts
  app/build.gradle.kts
  app/src/main/AndroidManifest.xml
  app/src/main/cpp/CMakeLists.txt
  app/src/main/res/drawable/localimage_icon.png
  .github/workflows/android-apk.yml
)
for f in "${required[@]}"; do
  test -f "$f" || { echo "MISSING: $f" >&2; exit 1; }
done
# Host test sources must remain tracked candidates.
test -f hostbuild/test_m10b_cpu.cpp
test -f hostbuild/test_m10b_graph.cpp
test -f hostbuild/test_m10b_transforms.cpp
# No unfinished native markers.
if grep -RniE 'TODO|FIXME|not implemented|fake|dummy|placeholder|UnsupportedOperationException' app/src/main/cpp --include='*.cpp' --include='*.h'; then
  echo 'Unfinished native marker found' >&2
  exit 1
fi
printf 'LocalImage project verification: PASS\n'
