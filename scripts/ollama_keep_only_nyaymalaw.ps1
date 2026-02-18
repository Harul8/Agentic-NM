# Keep only the two Ollama models used by Nyaymalaw; remove all others.
# Default: qwen2.5:7b-instruct, llama3.1:8b (match llm/config.py)
# Edit $KeepModels below if you use different names.
# Run: powershell -ExecutionPolicy Bypass -File scripts/ollama_keep_only_nyaymalaw.ps1

$KeepModels = @(
    "qwen2.5:7b-instruct",
    "llama3.1:8b"
)

Write-Host "Ollama models to KEEP: $($KeepModels -join ', ')"
Write-Host ""

$listOutput = ollama list 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: ollama list failed. Is Ollama running?" -ForegroundColor Red
    exit 1
}

$lines = $listOutput | Where-Object { $_.Trim() -ne "" } | Select-Object -Skip 1
$toRemove = @()
foreach ($line in $lines) {
    $name = ($line -split "\s+", 2)[0]
    if (-not $name) { continue }
    $keep = $false
    foreach ($k in $KeepModels) {
        if ($name -eq $k -or $name.StartsWith("$k")) {
            $keep = $true
            break
        }
    }
    if (-not $keep) {
        $toRemove += $name
    }
}

if ($toRemove.Count -eq 0) {
    Write-Host "No models to remove. Only the keep-list models are present."
    exit 0
}

Write-Host "Models that will be REMOVED:"
$toRemove | ForEach-Object { Write-Host "  - $_" }
Write-Host ""
$confirm = Read-Host "Proceed? (y/N)"
if ($confirm -ne "y" -and $confirm -ne "Y") {
    Write-Host "Cancelled."
    exit 0
}

foreach ($name in $toRemove) {
    Write-Host "Removing $name ..."
    ollama rm $name 2>&1
}
Write-Host "Done."
