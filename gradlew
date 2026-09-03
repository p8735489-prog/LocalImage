#!/bin/sh
set -eu
GRADLE_VERSION="${GRADLE_VERSION:-9.5.0}"
GRADLE_USER_HOME="${GRADLE_USER_HOME:-$HOME/.gradle}"
DIST_DIR="$GRADLE_USER_HOME/wrapper/dists/gradle-$GRADLE_VERSION-bin"
DIST_ZIP="$DIST_DIR/gradle-$GRADLE_VERSION-bin.zip"
INSTALL_DIR="$DIST_DIR/gradle-$GRADLE_VERSION"
if [ ! -x "$INSTALL_DIR/bin/gradle" ]; then
  mkdir -p "$DIST_DIR"
  if [ ! -f "$DIST_ZIP" ]; then
    command -v curl >/dev/null 2>&1 || { echo "curl is required to bootstrap Gradle" >&2; exit 1; }
    curl -fL --retry 3 --connect-timeout 15 -o "$DIST_ZIP.tmp" "https://services.gradle.org/distributions/gradle-$GRADLE_VERSION-bin.zip"
    mv "$DIST_ZIP.tmp" "$DIST_ZIP"
  fi
  command -v unzip >/dev/null 2>&1 || { echo "unzip is required to bootstrap Gradle" >&2; exit 1; }
  rm -rf "$INSTALL_DIR.tmp"
  mkdir -p "$INSTALL_DIR.tmp"
  unzip -q "$DIST_ZIP" -d "$INSTALL_DIR.tmp"
  mv "$INSTALL_DIR.tmp/gradle-$GRADLE_VERSION" "$INSTALL_DIR"
  rmdir "$INSTALL_DIR.tmp"
fi
exec "$INSTALL_DIR/bin/gradle" "$@"
