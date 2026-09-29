::
:: Copyright (c) 2026 Intel Corporation
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::      http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.
::
:: Installs the tools needed to build MediaPipe with Bazel on Windows into
:: C:\opt: MSYS2 (bash, patch, unzip), Python, Bazel and OpenCV (built from
:: source). Visual Studio 2022 (or newer) with the C++ workload must already be
:: installed. OpenVINO is downloaded by Bazel itself (see WORKSPACE).
::
:: Usage (from cmd.exe, so the environment is kept afterwards):
::   call prepare_build_dependencies.bat [expunge]
::     expunge - 1 forces reinstallation of all dependencies, default 0
::
:: Version overrides via environment: OPENCV_VERSION, PYTHON_VERSION.
:: Behind a proxy, set http_proxy and https_proxy before running; they are honored by curl, git, pacman and gpg.
@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "expunge=0"
if "%~1"=="1" set "expunge=1"
echo [INFO] expunge=%expunge%

set "script_dir=%~dp0"
set "opt_install_dir=C:\opt"
if not defined OPENCV_VERSION set "OPENCV_VERSION=4.13.0"
if not defined PYTHON_VERSION set "PYTHON_VERSION=3.12.10"
set "msys_date=2024-07-27"
set "msys_file=msys2-base-x86_64-20240727.sfx.exe"
set "curl_exe=%SystemRoot%\System32\curl.exe"
set "tar_exe=%SystemRoot%\System32\tar.exe"

set "bazel_version="
for /f "usebackq delims=" %%v in ("%script_dir%.bazelversion") do if not defined bazel_version set "bazel_version=%%v"
if not defined bazel_version (
    echo [ERROR] Cannot read %script_dir%.bazelversion
    goto :error
)

if not exist "%opt_install_dir%" mkdir "%opt_install_dir%" || goto :error

call :find_msvc || goto :error
call :install_msys || goto :error
call :install_python || goto :error
call :install_bazel || goto :error
call :install_opencv || goto :error

echo.
echo [INFO] All build dependencies installed.
echo [INFO] Build the OpenVINO example with:
echo   bazel --output_user_root=C:/b build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/examples/desktop/openvino_classification:openvino_classification

:: Export the build environment to the calling cmd.exe session.
endlocal & (
    set "BAZEL_VS=%BAZEL_VS%"
    set "BAZEL_VC=%BAZEL_VS%\VC"
    set "BAZEL_SH=%opt_install_dir%\msys64\usr\bin\bash.exe"
    set "PYTHON_BIN_PATH=%python_path%\python.exe"
    set "PATH=%opt_install_dir%;%python_path%;%python_path%\Scripts;%opt_install_dir%\msys64\usr\bin;%PATH%"
)
exit /b 0

:error
echo [ERROR] Build dependencies setup failed.
endlocal
exit /b 1

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:find_msvc
set "BAZEL_VS="
set "vs_bt=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools"
if exist "%vs_bt%\VC\Tools\MSVC" set "BAZEL_VS=%vs_bt%"
set "vswhere=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not defined BAZEL_VS if exist "%vswhere%" (
    for /f "usebackq delims=" %%i in (`"!vswhere!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "BAZEL_VS=%%i"
)
if not defined BAZEL_VS (
    echo [ERROR] Visual Studio 2022 with the "Desktop development with C++" workload is required.
    echo [ERROR] Install it from https://visualstudio.microsoft.com/downloads/ and rerun this script.
    exit /b 1
)
echo [INFO] Using MSVC from: %BAZEL_VS%

set "cmake_exe=%BAZEL_VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not exist "%cmake_exe%" (
    for /f "delims=" %%i in ('where cmake 2^>nul') do if not exist "!cmake_exe!" set "cmake_exe=%%i"
)
if not exist "%cmake_exe%" (
    echo [ERROR] CMake not found. Install the "C++ CMake tools for Windows" VS component.
    exit /b 1
)
where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] git not found in PATH. Install Git for Windows.
    exit /b 1
)
exit /b 0

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Provides bash, patch and unzip used by Bazel repository rules and genrules.
:install_msys
set "msys_path=%opt_install_dir%\msys64"
set "bash_path=%msys_path%\usr\bin\bash.exe"
echo [INFO] Installing MSYS2 ...
if %expunge% EQU 1 if exist "%msys_path%" rmdir /S /Q "%msys_path%" || exit /b 1
if exist "%bash_path%" (
    echo [INFO] MSYS2 already installed in: %msys_path%
) else (
    if not exist "%opt_install_dir%\%msys_file%" (
        "%curl_exe%" -L --fail -o "%opt_install_dir%\%msys_file%" https://github.com/msys2/msys2-installer/releases/download/%msys_date%/%msys_file% || exit /b 1
    )
    "%opt_install_dir%\%msys_file%" -y -o"%opt_install_dir%" || exit /b 1
)
:: Initialize the keyring before the first login shell does it: gpg's dirmngr
:: ignores http_proxy by default, so key refreshes time out behind a proxy.
set "gnupg_dir=%msys_path%\etc\pacman.d\gnupg"
if not exist "%gnupg_dir%\trustdb.gpg" (
    "%bash_path%" -c "/usr/bin/pacman-key --init" || exit /b 1
)
findstr /x /c:"honor-http-proxy" "%gnupg_dir%\dirmngr.conf" >nul 2>&1 || (echo honor-http-proxy>>"%gnupg_dir%\dirmngr.conf")
if not exist "%gnupg_dir%\populated.done" (
    "%bash_path%" -c "export PATH=/usr/bin; gpgconf --homedir /etc/pacman.d/gnupg --kill dirmngr; pacman-key --populate msys2" || exit /b 1
    type nul > "%gnupg_dir%\populated.done"
)
"%bash_path%" -lc "pacman -S --noconfirm --needed patch unzip zip" || exit /b 1
:: gpg-agent/dirmngr would otherwise keep files in msys64 locked.
"%bash_path%" -c "/usr/bin/gpgconf --homedir /etc/pacman.d/gnupg --kill all" >nul 2>&1
echo [INFO] MSYS2 installed in: %msys_path%
exit /b 0

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:install_python
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do set "python_dir=Python%%a%%b"
set "python_path=%opt_install_dir%\%python_dir%"
set "python_installer=%opt_install_dir%\python-%PYTHON_VERSION%-amd64.exe"
echo [INFO] Installing Python %PYTHON_VERSION% ...
if %expunge% EQU 1 if exist "%python_path%\python.exe" (
    if exist "%python_installer%" start "" /wait "%python_installer%" /quiet /uninstall
    if exist "%python_path%" rmdir /S /Q "%python_path%"
)
if exist "%python_path%\python.exe" (
    echo [INFO] Python already installed in: %python_path%
) else (
    if not exist "%python_installer%" (
        "%curl_exe%" -L --fail -o "%python_installer%" https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe || exit /b 1
    )
    start "" /wait "%python_installer%" /quiet InstallAllUsers=0 TargetDir="%python_path%" PrependPath=0 Include_launcher=0 Include_test=0 Shortcuts=0 /log "%opt_install_dir%\python_install.log"
    if not exist "%python_path%\python.exe" (
        echo [ERROR] Python installation failed, see %opt_install_dir%\python_install.log
        echo [ERROR] If another Python %PYTHON_VERSION% is installed, run %python_installer% and choose "Repair" or uninstall it first.
        exit /b 1
    )
)
"%python_path%\python.exe" -m ensurepip || exit /b 1
"%python_path%\python.exe" -m pip install --upgrade pip || exit /b 1
"%python_path%\python.exe" -m pip install "numpy<3" || exit /b 1
:: TensorFlow's WORKSPACE detects the Python version by running "python3 --version".
copy /Y "%python_path%\python.exe" "%python_path%\python3.exe" >nul || exit /b 1
echo [INFO] Python installed in: %python_path%
exit /b 0

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:install_bazel
set "bazel_file=bazel-%bazel_version%-windows-x86_64.exe"
echo [INFO] Installing Bazel %bazel_version% ...
if %expunge% EQU 1 if exist "%opt_install_dir%\%bazel_file%" del /Q "%opt_install_dir%\%bazel_file%"
if not exist "%opt_install_dir%\%bazel_file%" (
    "%curl_exe%" -L --fail -o "%opt_install_dir%\%bazel_file%" https://github.com/bazelbuild/bazel/releases/download/%bazel_version%/%bazel_file% || exit /b 1
)
copy /Y "%opt_install_dir%\%bazel_file%" "%opt_install_dir%\bazel.exe" >nul || exit /b 1
echo [INFO] Bazel installed: %opt_install_dir%\bazel.exe
exit /b 0

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Installed to C:\opt\opencv_<ver>; C:\opt\opencv is a junction used by WORKSPACE.
:install_opencv
set "opencv_src=%opt_install_dir%\opencv_git_%OPENCV_VERSION%"
set "opencv_contrib_src=%opt_install_dir%\opencv_contrib_git_%OPENCV_VERSION%"
set "opencv_install=%opt_install_dir%\opencv_%OPENCV_VERSION%"
set "opencv_link=%opt_install_dir%\opencv"
echo [INFO] Installing OpenCV %OPENCV_VERSION% ...
if %expunge% EQU 1 if exist "%opencv_install%" rmdir /S /Q "%opencv_install%" || exit /b 1
if exist "%opencv_install%\include\opencv2\core.hpp" (
    echo [INFO] OpenCV already installed in: %opencv_install%
    goto :link_opencv
)
if exist "%opencv_src%" rmdir /S /Q "%opencv_src%" || exit /b 1
if exist "%opencv_contrib_src%" rmdir /S /Q "%opencv_contrib_src%" || exit /b 1
git clone --depth 1 --branch %OPENCV_VERSION% https://github.com/opencv/opencv.git "%opencv_src%" || exit /b 1
git clone --depth 1 --branch %OPENCV_VERSION% https://github.com/opencv/opencv_contrib.git "%opencv_contrib_src%" || exit /b 1

set "opencv_flags=-D BUILD_LIST=core,imgproc,imgcodecs,calib3d,features2d,highgui,video,videoio,optflow"
set "opencv_flags=%opencv_flags% -D BUILD_opencv_world=ON -D BUILD_SHARED_LIBS=ON"
set "opencv_flags=%opencv_flags% -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D BUILD_DOCS=OFF"
set "opencv_flags=%opencv_flags% -D BUILD_opencv_apps=OFF -D BUILD_opencv_python3=OFF -D BUILD_JAVA=OFF"
set "opencv_flags=%opencv_flags% -D WITH_FFMPEG=ON -D WITH_JPEG=ON -D WITH_PNG=ON -D WITH_TIFF=OFF -D WITH_OPENEXR=OFF"
set "opencv_flags=%opencv_flags% -D CV_ENABLE_INTRINSICS=ON -D WITH_EIGEN=OFF"

"%cmake_exe%" -S "%opencv_src%" -B "%opencv_src%\build" -A x64 -D CMAKE_INSTALL_PREFIX="%opencv_install%" -D OPENCV_EXTRA_MODULES_PATH="%opencv_contrib_src%\modules" %opencv_flags% || exit /b 1
"%cmake_exe%" --build "%opencv_src%\build" --config Release --target INSTALL --parallel %NUMBER_OF_PROCESSORS% || exit /b 1

:link_opencv
if exist "%opencv_link%" rmdir "%opencv_link%" || exit /b 1
mklink /J "%opencv_link%" "%opencv_install%" >nul || exit /b 1
echo [INFO] OpenCV %OPENCV_VERSION% installed in: %opencv_install% (linked as %opencv_link%)
exit /b 0
