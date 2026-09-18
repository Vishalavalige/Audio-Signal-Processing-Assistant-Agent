# Creates a desktop shortcut that launches the Audio Signal Processing Agent
$batPath    = Join-Path $PSScriptRoot "Launch Audio Agent.bat"
$shortcut   = Join-Path ([Environment]::GetFolderPath("Desktop")) "Audio Signal Agent.lnk"
$iconPath   = Join-Path $PSScriptRoot "app_icon.ico"

$WS = New-Object -ComObject WScript.Shell
$SC = $WS.CreateShortcut($shortcut)
$SC.TargetPath       = $batPath
$SC.WorkingDirectory = $PSScriptRoot
$SC.Description      = "Audio Signal Processing Agent Dashboard"
$SC.WindowStyle      = 1

# Use a built-in Windows icon if no custom icon exists
if (Test-Path $iconPath) {
    $SC.IconLocation = $iconPath
} else {
    $SC.IconLocation = "C:\Windows\System32\shell32.dll,14"
}

$SC.Save()
Write-Host ""
Write-Host "  Shortcut created on your Desktop:" -ForegroundColor Green
Write-Host "  $shortcut" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Double-click 'Audio Signal Agent' on your desktop to launch the app." -ForegroundColor Yellow
Write-Host ""
