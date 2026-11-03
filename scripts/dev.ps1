<#
.SYNOPSIS
    Daily developer shortcuts for TaxLens (Windows PowerShell 5.1+ / PowerShell 7).

.EXAMPLE
    .\scripts\dev.ps1 check      # lint + format check + types + tests (run before every commit)
    .\scripts\dev.ps1 serve      # API with reload on http://localhost:8000/docs
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "lint", "format", "types", "test", "check", "serve", "hooks", "help")]
    [string]$Task = "help"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Invoke-Step([string]$Name, [scriptblock]$Command) {
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

switch ($Task) {
    "setup" {
        Invoke-Step "uv sync" { uv sync }
        Invoke-Step "install git hooks" { uv run pre-commit install }
    }
    "lint"   { Invoke-Step "ruff check" { uv run ruff check . } }
    "format" { Invoke-Step "ruff format" { uv run ruff format . } }
    "types"  { Invoke-Step "mypy" { uv run mypy taxlens tests } }
    "test"   { Invoke-Step "pytest" { uv run pytest } }
    "check" {
        Invoke-Step "ruff check" { uv run ruff check . }
        Invoke-Step "ruff format --check" { uv run ruff format --check . }
        Invoke-Step "mypy" { uv run mypy taxlens tests }
        Invoke-Step "pytest" { uv run pytest }
        Write-Host "All checks passed." -ForegroundColor Green
    }
    "serve" { Invoke-Step "uvicorn" { uv run uvicorn taxlens.main:app --reload } }
    "hooks" { Invoke-Step "pre-commit (all files)" { uv run pre-commit run --all-files } }
    default {
        Write-Host "Usage: .\scripts\dev.ps1 <setup|lint|format|types|test|check|serve|hooks>"
    }
}
