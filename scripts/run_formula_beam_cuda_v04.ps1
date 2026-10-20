param(
    [ValidateRange(0,511)][int]$Start = 0,
    [ValidateRange(1,512)][int]$Count = 32,
    [ValidateSet(5,20,50)][int[]]$Beams = @(5,20,50),
    [string]$ExperimentRoot = 'outputs/formula_constrained_v04_rtx4070_20261020'
)
$ErrorActionPreference = 'Stop'
if ($Start + $Count -gt 512) { throw 'Slice exceeds the frozen 512-head cohort' }
. (Join-Path $PSScriptRoot 'enter_cuda_v04.ps1')
foreach ($beam in $Beams) {
    foreach ($encoder in @('fingerprint','graph')) {
        foreach ($mode in @('raw','elements','elements_grammar')) {
            $runName = "${encoder}_${mode}_b${beam}_s${Start}"
            & python scripts/supervise_formula_beam_cuda_v04.py `
                --encoder $encoder --mode $mode --beam-size $beam `
                --start $Start --count $Count --run-name $runName `
                --experiment-root $ExperimentRoot
            if ($LASTEXITCODE -ne 0) { throw "GPU work stopped at $runName; no automatic retry" }
        }
    }
}
