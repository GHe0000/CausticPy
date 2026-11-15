param([Parameter(ValueFromRemainingArguments=$true)][string[]]$DesignArgs)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
. "$PSScriptRoot\activate.ps1"
python -u "$PSScriptRoot\design_lens.py" @DesignArgs
exit $LASTEXITCODE
