$taskFolder = Split-Path -Parent $MyInvocation.MyCommand.Path
$taskPython = Join-Path $taskFolder 'venv\Scripts\pythonw.exe'
$taskScript = Join-Path $taskFolder 'live_transcriber.py'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'The Python environment is missing. See README.md for setup.'
}
Start-Process -FilePath $taskPython -ArgumentList ('"' + $taskScript + '"') -WorkingDirectory $taskFolder -WindowStyle Hidden
