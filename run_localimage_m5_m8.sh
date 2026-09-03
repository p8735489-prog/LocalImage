#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "LocalImage source audit"
grep -RniE 'TODO|FIXME|not implemented|fake|dummy' "$ROOT/app/src/main/cpp" "$ROOT/app/src/main/java" "$ROOT/app/src/main/kotlin" 2>/dev/null && {
  echo "ERROR: unfinished implementation marker found" >&2
  exit 1
} || true
test -f "$ROOT/gradlew"
test -f "$ROOT/app/src/main/cpp/CMakeLists.txt"
echo "Audit complete"
