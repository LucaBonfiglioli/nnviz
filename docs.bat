del /s /f /q docs\_static\generated\*.* >nul 2>&1
del /s /f /q docs\api\generated\*.* >nul 2>&1
sphinx-apidoc -o docs\api\generated nnviz
call docs\make.bat clean
call docs\make.bat html
if exist docs\_build\html\index.html start docs\_build\html\index.html