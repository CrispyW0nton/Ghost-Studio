Ghost Studio @VERSION@ for Windows (64-bit)
===========================================

Ghost Studio is a free, open-source modding suite for KOTOR 1 and KOTOR 2:
models, characters, animations, modules and maps, scripts, 2DA tables and
packaging. Source and docs: https://github.com/CrispyW0nton/Ghost-Studio

Install
-------
1. Unzip this folder anywhere, for example Documents\GhostStudio.
2. Double-click GhostStudio.exe.

Nothing else to install: Ghost Studio brings its own Python 3.13 and packages
in the "python" folder. Keep GhostStudio.exe in this folder with its DLLs and
the python folder.

If Windows says VCRUNTIME140.dll or MSVCP140.dll is missing, install the
Microsoft Visual C++ Redistributable (x64) from Microsoft:
https://aka.ms/vs/17/release/vc_redist.x64.exe

First use
---------
Open Settings > Game Paths and set your KOTOR 2 (and KOTOR 1) folder.
Making mods for Kotor Academy: Expanded Galaxy? Read
https://github.com/CrispyW0nton/Ghost-Studio/blob/main/docs/KJA_MODDERS_START_HERE.md

Use it with an AI agent (Claude, Codex, Cursor)
-----------------------------------------------
Ghost Studio's KotorMCP server runs on the bundled Python. Point your MCP
client at it, using this folder's full path (example: Documents\GhostStudio):

  command: C:\path\to\GhostStudio\python\python.exe
  args:    C:\path\to\GhostStudio\mcp\scripts\mcp\start_kotormcp_stdio.py
  env:     K2_PATH = your KOTOR 2 folder (and K1_PATH for KOTOR 1)

Point K2_PATH at a copy of the game while you experiment, and approve any tool
that installs files or drives a running game yourself.

License
-------
MIT. See LICENSE and NOTICE.
Ghost Studio does not include any KOTOR game files.
