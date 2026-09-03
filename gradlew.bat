@echo off
setlocal
if "%GRADLE_VERSION%"=="" set "GRADLE_VERSION=9.5.0"
if "%GRADLE_USER_HOME%"=="" set "GRADLE_USER_HOME=%USERPROFILE%\.gradle"
set "DIST_DIR=%GRADLE_USER_HOME%\wrapper\dists\gradle-%GRADLE_VERSION%-bin"
set "DIST_ZIP=%DIST_DIR%\gradle-%GRADLE_VERSION%-bin.zip"
set "INSTALL_DIR=%DIST_DIR%\gradle-%GRADLE_VERSION%"
if exist "%INSTALL_DIR%\bin\gradle.bat" goto run
if not exist "%DIST_DIR%" mkdir "%DIST_DIR%"
if not exist "%DIST_ZIP%" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri 'https://services.gradle.org/distributions/gradle-%GRADLE_VERSION%-bin.zip' -OutFile '%DIST_ZIP%.tmp'; Move-Item -Force '%DIST_ZIP%.tmp' '%DIST_ZIP%'"
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force '%DIST_ZIP%' '%DIST_DIR%\extract'"
move /Y "%DIST_DIR%\extract\gradle-%GRADLE_VERSION%" "%INSTALL_DIR%" >nul
:run
call "%INSTALL_DIR%\bin\gradle.bat" %*
exit /b %ERRORLEVEL%
