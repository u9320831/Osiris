@echo off
setlocal
cd /d "%~dp0"

docker compose --profile kafka -f compose.yaml stop qdrant kafka
if errorlevel 1 (
    echo.
    echo Echec de l'arret de Kafka et Qdrant.
    echo Verifiez que Docker Desktop est demarre.
    exit /b 1
)

echo.
echo Kafka et Qdrant sont arretes. Les volumes et les donnees sont conserves.
