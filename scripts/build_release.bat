@echo off
rem Nexus Release derlemesi: yapilandir, derle, test et.
rem Kullanim: scripts\build_release.bat [build-dizini]
rem
rem Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
rem 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
rem olarak gelistirilmistir.

setlocal
set "BUILD_DIR=%~1"
if "%BUILD_DIR%"=="" set "BUILD_DIR=build-phase1"
set "ROOT=%~dp0.."
set "CMAKE=C:\Program Files\CMake\bin\cmake.exe"
set "CTEST=C:\Program Files\CMake\bin\ctest.exe"

if not defined VCINSTALLDIR (
    for %%E in (Preview Community Professional Enterprise BuildTools) do (
        if exist "C:\Program Files\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat" (
            call "C:\Program Files\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat" >nul
            goto :have_vc
        )
    )
    echo vcvars64.bat bulunamadi
    exit /b 2
)
:have_vc

"%CMAKE%" -S "%ROOT%" -B "%ROOT%\%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release || exit /b 1
"%CMAKE%" --build "%ROOT%\%BUILD_DIR%" --parallel || exit /b 1
if /I "%2"=="notest" exit /b 0
"%CTEST%" --test-dir "%ROOT%\%BUILD_DIR%" --output-on-failure -C Release %3 %4 %5
exit /b %ERRORLEVEL%
