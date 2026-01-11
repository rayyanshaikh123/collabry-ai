"""Open applications or execute simple system tasks."""

import os
import subprocess
import shutil
from pathlib import Path

def _find_app_windows(app_name: str):
    """
    Dynamically find application on Windows using multiple strategies.
    NO HARDCODING - Pure dynamic discovery.
    """
    app_lower = app_name.lower().strip()
    
    # 1. Try PATH lookup first (fastest for command-line tools)
    for variant in [app_name, f"{app_name}.exe", f"{app_lower}.exe"]:
        path_result = shutil.which(variant)
        if path_result:
            return ('path', path_result)
    
    # 2. Search for Windows Store apps using PowerShell
    try:
        ps_cmd = f'''powershell -Command "Get-AppxPackage | Where-Object {{$_.Name -like '*{app_name}*'}} | Select-Object -First 1 -ExpandProperty PackageFamilyName"'''
        result = subprocess.run(ps_cmd, capture_output=True, text=True, shell=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            package_family = result.stdout.strip()
            # Get the app ID
            ps_cmd2 = f'''powershell -Command "(Get-StartApps | Where-Object {{$_.AppId -like '*{package_family}*'}}).AppId"'''
            result2 = subprocess.run(ps_cmd2, capture_output=True, text=True, shell=True, timeout=5)
            if result2.returncode == 0 and result2.stdout.strip():
                return ('store', result2.stdout.strip())
            # Fallback to package family name
            return ('store', f'{package_family}!App')
    except:
        pass
    
    # 3. Use PowerShell Get-Command to find installed desktop apps
    try:
        ps_cmd = f'powershell -Command "Get-Command {app_name} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source"'
        result = subprocess.run(ps_cmd, capture_output=True, text=True, shell=True, timeout=3)
        if result.returncode == 0 and result.stdout.strip():
            app_path = result.stdout.strip()
            if Path(app_path).exists():
                return ('path', app_path)
    except:
        pass
    
    # 4. Search Program Files directories (smart, limited depth)
    search_dirs = [
        Path(os.environ.get('ProgramFiles', 'C:\\Program Files')),
        Path(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)')),
        Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs',
    ]
    
    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
        try:
            # Look for matching folder name first
            for folder in base_dir.iterdir():
                if not folder.is_dir():
                    continue
                if app_lower in folder.name.lower():
                    # Found matching folder, search for main exe
                    for exe_file in folder.glob("*.exe"):
                        if app_lower in exe_file.stem.lower():
                            return ('path', str(exe_file))
                    # Return first exe if no exact match
                    exes = list(folder.glob("*.exe"))
                    if exes:
                        return ('path', str(exes[0]))
        except (PermissionError, OSError):
            continue
    
    # 5. Return as-is - try direct launch
    return ('direct', app_name)

def open_app(app: str):
    """
    Open a local application dynamically without hardcoding.
    Searches system for the app and launches it.
    
    Args:
        app: Application name (e.g., 'notepad', 'chrome', 'vscode', 'netflix')
    """
    try:
        if os.name == "nt":  # Windows
            app_type, app_path = _find_app_windows(app)
            
            if app_type == 'store':
                # Windows Store app - use explorer.exe shell:AppsFolder
                subprocess.Popen(
                    f'explorer.exe shell:AppsFolder\\{app_path}',
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                return {
                    "message": f"Opened Store app: {app}",
                    "app_id": app_path,
                    "status": "success"
                }
            elif app_type == 'path':
                # Desktop app with file path
                try:
                    os.startfile(app_path)
                except OSError:
                    subprocess.Popen(
                        f'start "" "{app_path}"',
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                return {
                    "message": f"Opened app: {app}",
                    "path": app_path,
                    "status": "success"
                }
            else:
                # Direct launch attempt (protocols, shortcuts, etc.)
                subprocess.Popen(
                    f'start "" "{app_path}"',
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                return {
                    "message": f"Attempted to open: {app}",
                    "status": "success"
                }
        else:  # Linux / Mac
            # Try which command to find app
            app_path = shutil.which(app)
            if app_path:
                subprocess.Popen([app_path])
                return {"message": f"Opened app: {app}", "path": app_path}
            else:
                subprocess.Popen([app])
                return {"message": f"Opened app: {app}"}
                
    except Exception as e:
        return {"error": f"Failed to open {app}: {str(e)}"}

TOOL = {
    "name": "open_app",
    "description": "Open any local application dynamically by name (e.g., notepad, chrome, vscode, calculator). Searches system automatically.",
    "func": open_app
}
