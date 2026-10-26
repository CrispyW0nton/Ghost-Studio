"""Discoverable native editor action over the same unattended exporter."""
from pathlib import Path
from uuid import uuid4
import unreal
import json
import os
import subprocess
from ghoststudio_unreal_exporter import export_level


@unreal.uclass()
class GhostStudioExportMenu(unreal.ToolMenuEntryScript):
    @unreal.ufunction(override=True)
    def execute(self, context):
        try:
            destination = Path(unreal.Paths.project_saved_dir()) / "GhostStudio" / ("export-" + uuid4().hex)
            result = export_level(destination)
            connection = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "GhostStudio/unreal-connection.json"
            if connection.is_file():
                settings = json.loads(connection.read_text(encoding="utf-8"))
                executable = Path(settings.get("ghost_executable", ""))
                if settings.get("schema") == "ghoststudio.unreal-connection/v1" and executable.is_file() and executable.name.lower() == "ghoststudio.exe":
                    subprocess.Popen([str(executable), "--unreal-snapshot", str(result)], cwd=executable.parent)
                    return
            unreal.EditorDialog.show_message("Ghost Studio snapshot exported", f"Open this snapshot in Map Studio:\n{result}\n\nThis is source data. Review conversion and collision choices before building.", unreal.AppMsgType.OK)
        except Exception as error:
            unreal.EditorDialog.show_message("Ghost Studio export needs attention", str(error), unreal.AppMsgType.OK)


_entry = GhostStudioExportMenu()
_entry.init_entry("GhostStudioExporter", "LevelEditor.MainMenu.Tools", "GhostStudio", "GhostStudio.Export", "Send Level to Ghost Studio", "Export a preserved native snapshot and open the import review in Ghost Studio.")
_entry.register_menu_entry()
