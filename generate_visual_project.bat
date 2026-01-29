@echo off

echo Creating DenseMatrixCuda Project...
cmake -S . -B build -G "Visual Studio 16 2019" -A x64

pause