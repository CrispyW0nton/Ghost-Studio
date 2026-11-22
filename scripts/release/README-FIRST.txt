Ghost Studio @VERSION@ for Windows (64-bit)
===========================================

Ghost Studio is a free, open-source modding suite for KOTOR 1 and KOTOR 2:
models, characters, animations, modules and maps, scripts, 2DA tables and
packaging. Source and docs: https://github.com/CrispyW0nton/Ghost-Studio

Install
-------
1. Install 64-bit Python 3.13 from https://www.python.org/downloads/
   (Windows installer). Keep the default install location.
2. Unzip this folder anywhere, for example Documents\GhostStudio.
3. Double-click setup.bat once. It installs the Python packages Ghost Studio
   needs.
4. Double-click GhostStudio.exe.

Keep GhostStudio.exe in this folder with its DLLs.

If Windows says VCRUNTIME140.dll or MSVCP140.dll is missing, install the
Microsoft Visual C++ Redistributable (x64) from Microsoft:
https://aka.ms/vs/17/release/vc_redist.x64.exe

If your Python 3.13 is not in the default location, set the environment
variable GHOSTRIGGER_PYTHON_HOME to that folder (the one holding python313.dll
and Lib).

First use
---------
Open Settings > Game Paths and set your KOTOR 2 (and KOTOR 1) folder.
Making mods for Kotor Academy: Expanded Galaxy? Read
https://github.com/CrispyW0nton/Ghost-Studio/blob/main/docs/KJA_MODDERS_START_HERE.md

License
-------
MIT. See LICENSE and NOTICE.
Ghost Studio does not include any KOTOR game files.
