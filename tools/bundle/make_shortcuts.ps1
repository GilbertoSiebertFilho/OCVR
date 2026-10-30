# Desktop shortcuts for the Olds College apps bundle: OCVR, OCCD and NDVI Map Viewer (called by INSTALAR.bat).
param([string]$Root = (Split-Path -Parent $PSScriptRoot), [string]$Desktop = '')
$Root = (Resolve-Path $Root).Path
$desk = if ($Desktop) { $Desktop } else { [Environment]::GetFolderPath('Desktop') }
$sh = New-Object -ComObject WScript.Shell
$icon = Join-Path $Root 'tools\oc.ico'
$apps = @(
  @{ name = 'OCVR - taxa variavel'; bat = 'OCVR.bat' },
  @{ name = 'OCCD - limpeza de colheita'; bat = 'OCCD.bat' },
  @{ name = 'NDVI Map Viewer'; bat = 'NDVI.bat' }
)
foreach ($a in $apps) {
  $lnk = $sh.CreateShortcut((Join-Path $desk ($a.name + '.lnk')))
  $lnk.TargetPath = Join-Path $Root $a.bat
  $lnk.WorkingDirectory = $Root
  if (Test-Path $icon) { $lnk.IconLocation = $icon }
  $lnk.Save()
  Write-Host ('  atalho: ' + $a.name)
}
