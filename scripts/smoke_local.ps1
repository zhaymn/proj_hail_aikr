param(
    [Parameter(Mandatory = $true)]
    [string]$PdfPath,
    [string]$ApiBaseUrl = "http://127.0.0.1:8010",
    [string]$SessionId = ("smoke-local-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [switch]$ShowAnswerPreview
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=== " + $Text + " ===") -ForegroundColor Cyan
}

function Invoke-JsonPost {
    param(
        [string]$Url,
        [hashtable]$Body,
        [int]$TimeoutSec = 240
    )
    $payload = $Body | ConvertTo-Json -Depth 10
    return Invoke-RestMethod -Uri $Url -Method Post -ContentType "application/json" -Body $payload -TimeoutSec $TimeoutSec
}

function Upload-PdfMultipart {
    param(
        [string]$Url,
        [string]$Path
    )
    Add-Type -AssemblyName System.Net.Http

    $client = [System.Net.Http.HttpClient]::new()
    $fileStream = $null
    $form = $null
    $streamContent = $null

    try {
        $client.Timeout = [TimeSpan]::FromSeconds(300)

        $fileStream = [System.IO.File]::OpenRead($Path)
        $streamContent = [System.Net.Http.StreamContent]::new($fileStream)
        $streamContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse("application/pdf")

        $form = [System.Net.Http.MultipartFormDataContent]::new()
        $fileName = [System.IO.Path]::GetFileName($Path)
        $form.Add($streamContent, "files", $fileName)

        $response = $client.PostAsync($Url, $form).GetAwaiter().GetResult()
        $raw = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()

        if (-not $response.IsSuccessStatusCode) {
            throw "Upload failed with status $($response.StatusCode): $raw"
        }
        return $raw | ConvertFrom-Json
    }
    finally {
        if ($streamContent) { $streamContent.Dispose() }
        if ($form) { $form.Dispose() }
        if ($fileStream) { $fileStream.Dispose() }
        $client.Dispose()
    }
}

function New-ResultRow {
    param(
        [string]$Id,
        [string]$Query,
        [bool]$Pass,
        [string]$Reason,
        [string]$Provider,
        [bool]$ModelFallback,
        [int]$CitationCount,
        [string]$ResponseStage,
        [string]$ValidationStage,
        [string]$Answer
    )

    [pscustomobject]@{
        id = $Id
        query = $Query
        pass = $Pass
        reason = $Reason
        provider = $Provider
        model_fallback = $ModelFallback
        citations = $CitationCount
        response_stage = $ResponseStage
        validation_stage = $ValidationStage
        answer_preview = $Answer
    }
}

function Safe-String {
    param([object]$Value)
    if ($null -eq $Value) {
        return ""
    }
    return [string]$Value
}

function Evaluate-Case {
    param(
        [string]$CaseId,
        [string]$Answer,
        [int]$CitationCount
    )

    $text = (Safe-String $Answer).Trim()
    $lower = $text.ToLowerInvariant()

    switch ($CaseId) {
        "Q1" {
            $hasAtLeast6 = $lower.Contains("at least 6")
            $hasExpert6 = $lower.Contains("expert 6")
            $hasSixExperts = $lower.Contains("6 experts")
            if ($hasAtLeast6 -or $hasExpert6 -or $hasSixExperts) { return @{ pass = $true; reason = "Expert count extracted." } }
            return @{ pass = $false; reason = "Expected explicit expert count around 6." }
        }
        "Q2" {
            $okTopK = $lower.Contains("top-k") -or $lower.Contains("top k")
            $okShared = $lower.Contains("shared expert")
            $okTwo = $text -match "\b2\b"
            if ($okTopK -and $okShared -and $okTwo) { return @{ pass = $true; reason = "Default config values present." } }
            return @{ pass = $false; reason = "Expected Top-K and shared expert defaults (2,2)." }
        }
        "Q3" {
            $hasPretrain = $lower.Contains("pretrain")
            $hasAlpha = $lower.Contains("alpha") -or $text.Contains("\alpha") -or $lower.Contains("10^-4") -or $lower.Contains("10−4") -or $lower.Contains("10 -4")
            $hasAux = $lower.Contains("aux") -or $lower.Contains("auxiliary loss")
            if ($hasPretrain -and $hasAlpha -and $hasAux -and $CitationCount -ge 1) {
                return @{ pass = $true; reason = "Pretraining math and alpha explained." }
            }
            return @{ pass = $false; reason = "Expected L_pretrain, alpha, and L_aux explanation with citation." }
        }
        "Q4" {
            if ($text -eq "This information is not in your uploaded papers.") {
                return @{ pass = $true; reason = "Correct abstention for missing detail." }
            }
            return @{ pass = $false; reason = "Expected strict abstention for unavailable transformer-head detail." }
        }
        "Q5" {
            $hasDi = $lower.Contains("d_i") -or $text -match "\bDi\b"
            $hasLaux = $lower.Contains("laux") -or $lower.Contains("auxiliary loss") -or $lower.Contains("l_aux")
            $hasBatch = $lower.Contains("batch") -or $lower.Contains("token")
            if ($hasDi -and $hasLaux -and $hasBatch -and $CitationCount -ge 1) {
                return @{ pass = $true; reason = "Routing/batch math covered." }
            }
            return @{ pass = $false; reason = "Expected D_i, L_aux, and batch computation details." }
        }
        "Q6" {
            $hasRouting = $lower.Contains("routing") -or $lower.Contains("top-k") -or $lower.Contains("expert")
            $hasLoss = $lower.Contains("loss") -or $lower.Contains("laux") -or $lower.Contains("pretrain")
            if ($hasRouting -and $hasLoss -and $CitationCount -ge 2) {
                return @{ pass = $true; reason = "Long-form math answer is grounded." }
            }
            return @{ pass = $false; reason = "Expected end-to-end math explanation with citations." }
        }
        default {
            return @{ pass = $true; reason = "No check for case." }
        }
    }
}

if (-not (Test-Path -LiteralPath $PdfPath)) {
    throw "PDF not found: $PdfPath"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$reportDir = Join-Path $repoRoot "docs\smoke_reports"
if (-not (Test-Path -LiteralPath $reportDir)) {
    New-Item -ItemType Directory -Path $reportDir | Out-Null
}

$healthUrl = "$ApiBaseUrl/health"
$uploadUrl = "$ApiBaseUrl/api/papers/upload"
$chatUrl = "$ApiBaseUrl/api/chat"

Write-Section "Health Check"
$health = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 10
Write-Host ("status=" + $health.status + " llm_available=" + $health.llm_available + " indexed_papers=" + $health.indexed_papers)
if ($health.status -ne "ok") {
    throw "Backend health check failed."
}

Write-Section "Upload PDF"
$uploadResponse = Upload-PdfMultipart -Url $uploadUrl -Path $PdfPath
if (-not $uploadResponse.papers -or $uploadResponse.papers.Count -lt 1) {
    throw "Upload succeeded but did not return paper metadata."
}
$paper = $uploadResponse.papers[0]
$paperId = [string]$paper.paper_id
Write-Host ("uploaded paper_id=" + $paperId + " filename=" + $paper.filename)

$tests = @(
    @{ id = "Q1"; query = "how many experts were used in mixture of experts" },
    @{ id = "Q2"; query = "in default configuration what are Top-K and shared experts values" },
    @{ id = "Q3"; query = "write the pretraining objective math with equations and explain each symbol including auxiliary loss alpha value" },
    @{ id = "Q4"; query = "what was number of transformer heads used" },
    @{ id = "Q5"; query = "explain the math of routing probabilities and how Di and Laux are computed from a batch" },
    @{ id = "Q6"; query = "teach me the complete math of this paper from token routing to final loss in detail using only this paper" }
)

$results = @()
Write-Section "Run Local Brain Tests"
foreach ($test in $tests) {
    Write-Host ("[" + $test.id + "] " + $test.query)
    $body = @{
        session_id = $SessionId
        mode = "local"
        message = $test.query
        paper_ids = @($paperId)
        history = @()
    }

    try {
        $response = Invoke-JsonPost -Url $chatUrl -Body $body -TimeoutSec 300
        $answer = Safe-String $response.answer
        $debug = $response.debug
        $citations = @($response.citations)
        $citationCount = $citations.Count
        $provider = Safe-String $debug.model_provider
        $modelFallback = [bool]($debug.model_fallback)
        $responseStage = Safe-String $debug.response_stage
        $validationStage = Safe-String $debug.validation_stage

        $evaluation = Evaluate-Case -CaseId $test.id -Answer $answer -CitationCount $citationCount

        $hardChecksPass = $true
        $hardReason = @()
        if ($test.id -ne "Q4" -and $modelFallback) {
            $hardChecksPass = $false
            $hardReason += "Unexpected model fallback."
        }
        if ($test.id -ne "Q4" -and [string]::IsNullOrWhiteSpace($provider)) {
            $hardChecksPass = $false
            $hardReason += "Missing model provider."
        }
        if ($responseStage -ne "finalized") {
            $hardChecksPass = $false
            $hardReason += "Unexpected response_stage=$responseStage."
        }

        $pass = $evaluation.pass -and $hardChecksPass
        $reason = $evaluation.reason
        if (-not $hardChecksPass) {
            $reason = ($reason + " " + ($hardReason -join " ")).Trim()
        }

        $preview = $answer -replace "\s+", " "
        if ($preview.Length -gt 220) {
            $preview = $preview.Substring(0, 220) + "..."
        }

        $results += New-ResultRow `
            -Id $test.id `
            -Query $test.query `
            -Pass $pass `
            -Reason $reason `
            -Provider $provider `
            -ModelFallback $modelFallback `
            -CitationCount $citationCount `
            -ResponseStage $responseStage `
            -ValidationStage $validationStage `
            -Answer $preview

        if ($ShowAnswerPreview) {
            Write-Host ("  -> " + $preview)
        }
    }
    catch {
        $results += New-ResultRow `
            -Id $test.id `
            -Query $test.query `
            -Pass $false `
            -Reason ("Request failed: " + $_.Exception.Message) `
            -Provider "" `
            -ModelFallback $true `
            -CitationCount 0 `
            -ResponseStage "request_error" `
            -ValidationStage "" `
            -Answer ""
    }
}

Write-Section "Summary"
$results | Format-Table id, pass, provider, model_fallback, citations, response_stage, validation_stage, reason -AutoSize

$overallPass = (@($results | Where-Object { -not $_.pass }).Count -eq 0)
$providerPass = (@($results | Where-Object { $_.provider -eq "openrouter" -or $_.provider -eq "groq" -or $_.provider -eq "gemini" }).Count -ge 1)
if (-not $providerPass) {
    $overallPass = $false
}

$report = [pscustomobject]@{
    timestamp = (Get-Date).ToString("o")
    api_base = $ApiBaseUrl
    session_id = $SessionId
    paper_id = $paperId
    filename = $paper.filename
    llm_available = [bool]$health.llm_available
    overall_pass = $overallPass
    provider_seen = $providerPass
    results = $results
}

$reportPath = Join-Path $reportDir ("smoke_local_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".json")
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host ("report saved: " + $reportPath)

if ($overallPass) {
    Write-Host "SMOKE TEST: PASS" -ForegroundColor Green
    exit 0
}

Write-Host "SMOKE TEST: FAIL" -ForegroundColor Red
exit 1
