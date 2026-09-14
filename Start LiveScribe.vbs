Option Explicit
Dim shell, files, folder, python, script
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
folder = files.GetParentFolderName(WScript.ScriptFullName)
python = files.BuildPath(folder, "venv\Scripts\pythonw.exe")
script = files.BuildPath(folder, "live_transcriber.py")
If Not files.FileExists(python) Then
    MsgBox "The Python environment is missing. Follow the setup instructions in README.md.", 48, "LiveScribe"
    WScript.Quit 1
End If
shell.CurrentDirectory = folder
shell.Run Chr(34) & python & Chr(34) & " " & Chr(34) & script & Chr(34), 0, False
