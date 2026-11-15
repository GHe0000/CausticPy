param([string]$Target = 'target_saturn.png', [Parameter(Mandatory=$true)][string]$Out, [string]$Workflow = 'configs/workflow.json')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
. "$PSScriptRoot\activate.ps1"
$plan = Get-Content -LiteralPath $Workflow -Raw -Encoding utf8 | ConvertFrom-Json
python run.py --target $Target --config $plan.base_config --out "$Out/base"
if ($LASTEXITCODE -ne 0) { throw 'Base design failed; see its status.json.' }
$previous = "$Out/base"
foreach ($step in $plan.refinements) {
    $destination = "$Out/$($step.name)"
    $extra = @()
    if ($null -ne $step.mapping_anchor) { $extra = @("--mapping-anchor", [string]$step.mapping_anchor) }
    python refine.py $previous --resolution $step.resolution --iterations $step.iterations --thin-weight $step.thin_weight --adaptive $step.adaptive @extra --out $destination
    if ($LASTEXITCODE -ne 0) { throw "Refinement $($step.name) failed; see its status.json." }
    python audit.py $destination
    if ($LASTEXITCODE -ne 0) { throw 'Quality audit failed.' }
    $previous = $destination
}
