param(
    [Parameter(Mandatory = $true)]
    [string]$Text,
    [Parameter(Mandatory = $true)]
    [string]$ModelDir,
    [string]$OutputDir = '',
    [string]$ComputePython = '',
    [string]$TokenizerPython = '',
    [string]$Int8Dir = ''
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $OutputDir) { $OutputDir = Join-Path $root 'out\embedding' }
if (-not $ComputePython) { $ComputePython = (Get-Command python -ErrorAction Stop).Source }
if (-not $TokenizerPython) { $TokenizerPython = $ComputePython }

foreach ($path in @((Join-Path $ModelDir 'config.json'),
                    (Join-Path $ModelDir 'tokenizer.json'))) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required file: $path" }
}
if (-not $Int8Dir) {
    $indexPath = Join-Path $ModelDir 'model.safetensors.index.json'
    if (-not (Test-Path -LiteralPath $indexPath)) {
        throw "Missing weight index: $indexPath"
    }
    $index = Get-Content -LiteralPath $indexPath -Raw | ConvertFrom-Json
    $shardName = $index.weight_map.'model.embed_tokens.weight'
    if ([string]::IsNullOrWhiteSpace($shardName)) {
        throw "Embedding tensor is missing from weight index: $indexPath"
    }
    $shard = Join-Path $ModelDir $shardName
    if (-not (Test-Path -LiteralPath $shard)) {
        throw "Weight shard download is not complete: $shard"
    }
} elseif (-not (Test-Path -LiteralPath (Join-Path $Int8Dir 'quantization_manifest.json'))) {
    throw "INT8 table is not ready: $Int8Dir"
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$idsPath = Join-Path $OutputDir 'token_ids.json'
& $TokenizerPython (Join-Path $root 'python\tokenize_llama3.py') `
    --model-dir $ModelDir --text $Text --output $idsPath
if ($LASTEXITCODE -ne 0) { throw 'Tokenization failed' }

$embedArgs = @((Join-Path $root 'python\prepare_llama3_embedding.py'),
               '--model-dir', $ModelDir, '--token-ids-file', $idsPath,
               '--output-dir', $OutputDir)
$verifyArgs = @((Join-Path $root 'python\verify_llama3_embedding.py'),
                '--model-dir', $ModelDir, '--embedding-dir', $OutputDir)
if ($Int8Dir) {
    $embedArgs += @('--int8-dir', $Int8Dir)
    $verifyArgs += @('--int8-dir', $Int8Dir)
}
& $ComputePython @embedArgs
if ($LASTEXITCODE -ne 0) { throw 'Embedding generation failed' }
& $ComputePython @verifyArgs
if ($LASTEXITCODE -ne 0) { throw 'Embedding verification failed' }

Write-Host "Completed: $OutputDir"
