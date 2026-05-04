$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $projectRoot

uvicorn Code.Operationalization.secop_fastapi.main:app `
  --reload `
  --reload-dir Code/Operationalization/secop_fastapi `
  --host 127.0.0.1 `
  --port 8000
