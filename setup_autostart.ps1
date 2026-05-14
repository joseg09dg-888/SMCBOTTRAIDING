#Requires -RunAsAdministrator
<#
.SYNOPSIS
  Registra el SMC Trading Bot en Windows Task Scheduler para autoarranque.
  Ejecutar UNA SOLA VEZ como Administrador:
      powershell -ExecutionPolicy Bypass -File setup_autostart.ps1

  Para eliminar la tarea:
      Unregister-ScheduledTask -TaskName "SMC-TradingBot" -Confirm:$false
#>

$TaskName   = "SMC-TradingBot"
$BotDir     = "C:\Users\jose-\projects\trading_agent"
$BatFile    = "$BotDir\start_bot.bat"
$LogFile    = "$BotDir\logs\scheduler.log"
$TaskUser   = "SYSTEM"   # Corre aunque no hayas hecho login

# ── Validaciones previas ─────────────────────────────────────────────────────

if (-not (Test-Path $BatFile)) {
    Write-Error "No se encuentra $BatFile. Ejecuta este script desde el directorio del proyecto."
    exit 1
}

if (-not (Test-Path "$BotDir\logs")) {
    New-Item -ItemType Directory -Path "$BotDir\logs" -Force | Out-Null
}

# ── Eliminar tarea previa si existe ─────────────────────────────────────────

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "[OK] Tarea anterior eliminada."
}

# ── Definir acción ───────────────────────────────────────────────────────────

$Action = New-ScheduledTaskAction `
    -Execute  "cmd.exe" `
    -Argument "/c `"$BatFile`"" `
    -WorkingDirectory $BotDir

# ── Trigger: arranque de Windows ─────────────────────────────────────────────

$TriggerBoot = New-ScheduledTaskTrigger -AtStartup

# ── Configuración de la tarea ────────────────────────────────────────────────
# RestartCount / RestartInterval: reintentar hasta 3 veces cada 1 minuto si falla

$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit      (New-TimeSpan -Days 365) `
    -RestartCount            99 `
    -RestartInterval         (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable      `
    -RunOnlyIfNetworkAvailable

# ── Registrar tarea ──────────────────────────────────────────────────────────

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action   $Action `
    -Trigger  $TriggerBoot `
    -Settings $Settings `
    -RunLevel Highest `
    -User     $TaskUser `
    -Force | Out-Null

Write-Host ""
Write-Host "============================================================"
Write-Host "  SMC TradingBot — Task Scheduler configurado correctamente"
Write-Host "============================================================"
Write-Host "  Tarea:       $TaskName"
Write-Host "  Trigger:     Arranque de Windows"
Write-Host "  Reintentos:  Cada 1 minuto si el proceso muere"
Write-Host "  Usuario:     $TaskUser (sin sesion activa)"
Write-Host "  Log:         $LogFile"
Write-Host "  BAT:         $BatFile"
Write-Host ""
Write-Host "  Para iniciar ahora sin reiniciar:"
Write-Host "    Start-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
Write-Host "  Para ver estado:"
Write-Host "    Get-ScheduledTask -TaskName '$TaskName' | Select-Object State"
Write-Host ""
Write-Host "  Para eliminar:"
Write-Host "    Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
Write-Host "============================================================"

# ── Verificar que quedó registrada ──────────────────────────────────────────

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($task) {
    Write-Host "[OK] Tarea verificada: Estado = $($task.State)"
} else {
    Write-Error "[ERROR] La tarea no se registró correctamente."
    exit 1
}
