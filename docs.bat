del /s /f /q docs\_static\generated\*.* >nul 2>&1
del /s /f /q docs\api\generated\*.* >nul 2>&1
sphinx-apidoc -d 1 -o docs\api\generated nnviz
del /s /f /q docs\api\generated\modules.rst >nul 2>&1
call docs\make.bat clean
call docs\make.bat html
@REM if exist docs\_build\html\index.html start docs\_build\html\index.html