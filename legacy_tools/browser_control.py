"""Browser control tool for opening URLs, YouTube videos, Gmail, etc."""

import webbrowser
import subprocess
import urllib.parse
import requests
import re
from typing import Dict, Any

def browser_control(action: str = None, query: str = None, url: str = None, browser: str = None, target: str = None) -> Dict[str, Any]:
    """
    Control browser actions dynamically without hardcoding.
    
    Args:
        action: Type of action - 'open_url', 'youtube', 'gmail', 'search', 'pause', 'close' (auto-inferred if not provided)
        query: Search query or video name (for youtube/search)
        url: Direct URL to open (for open_url)
        browser: Browser name (chrome, firefox, edge) - optional, uses default if not specified
        target: For close action - specific tab name (e.g., 'youtube', 'chatgpt') or 'all'
    """
    try:
        # Auto-infer action if not provided - check all params for clues
        if not action:
            # If no params at all, this is likely a pause or close request
            # Default to pause as it's the most common media control
            if not query and not url and not target:
                action = 'pause'
            elif url:
                action = 'open_url'
            elif query:
                # Check if query looks like a YouTube request
                if any(word in query.lower() for word in ['play', 'video', 'song', 'music', 'youtube']):
                    action = 'youtube'
                else:
                    action = 'search'
            else:
                action = 'pause'  # Default fallback
        
        # Get browser controller
        browser_obj = None
        if browser:
            browser_lower = browser.lower()
            if 'chrome' in browser_lower:
                try:
                    browser_obj = webbrowser.get('chrome')
                except:
                    browser_obj = webbrowser.get()
            elif 'firefox' in browser_lower:
                try:
                    browser_obj = webbrowser.get('firefox')
                except:
                    browser_obj = webbrowser.get()
            elif 'edge' in browser_lower:
                try:
                    browser_obj = webbrowser.get('windows-default')
                except:
                    browser_obj = webbrowser.get()
        
        if not browser_obj:
            browser_obj = webbrowser.get()
        
        # Normalize action name (handle variations)
        action_lower = action.lower() if action else ''
        
        # Handle different actions
        if action_lower in ['open_url', 'open', 'browse'] and url:
            # Direct URL opening
            browser_obj.open(url)
            browser_name = browser if browser else "your default browser"
            return f"Successfully opened {url} in {browser_name}."
        
        elif action_lower in ['youtube', 'play', 'play_video', 'play_youtube'] and query:
            # Try to get the first video ID from YouTube search (no API key needed)
            try:
                search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(search_url, headers=headers, timeout=5)
                
                # Extract video ID from the HTML using regex
                # Look for videoId pattern in the page
                video_id_match = re.search(r'"videoId":"([a-zA-Z0-9_-]{11})"', response.text)
                
                if video_id_match:
                    video_id = video_id_match.group(1)
                    watch_url = f"https://www.youtube.com/watch?v={video_id}"
                    browser_obj.open(watch_url)
                    return f"Successfully opened and playing '{query}' on YouTube in your browser."
                else:
                    # Fallback to search results if no video ID found
                    browser_obj.open(search_url)
                    return f"Opened YouTube search results for '{query}'. Click the first video to play it."
            except:
                # If scraping fails, open search results page
                search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
                browser_obj.open(search_url)
                return f"Opened YouTube search for '{query}' in your browser."
        
        elif action_lower in ['gmail', 'email', 'mail']:
            # Open Gmail
            gmail_url = "https://mail.google.com"
            browser_obj.open(gmail_url)
            browser_name = browser if browser else "your browser"
            return f"Successfully opened Gmail in {browser_name}."
        
        elif action_lower in ['search', 'google'] and query:
            # Web search
            search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            browser_obj.open(search_url)
            return f"Opened Google search results for '{query}' in your browser."
        
        elif action_lower in ['pause', 'play', 'pause_play', 'toggle']:
            # Pause/Play media (sends Space key to active window)
            try:
                import pyautogui
                import time
                import subprocess
                
                # Try to bring browser to foreground
                try:
                    subprocess.run(
                        ['powershell', '-Command', 
                         '(New-Object -ComObject WScript.Shell).AppActivate((Get-Process chrome,firefox,msedge -ErrorAction SilentlyContinue | Select-Object -First 1).MainWindowTitle)'],
                        capture_output=True,
                        timeout=1
                    )
                except:
                    pass
                
                # Add delay for window to focus
                time.sleep(0.5)
                pyautogui.press('space')
                time.sleep(0.1)
                
                return "Toggled pause/play on the active media."
            except ImportError:
                return {"error": "pyautogui not installed. Run: pip install pyautogui"}
            except Exception as e:
                return f"Failed to toggle pause/play: {str(e)}"
        
        elif action_lower in ['close', 'close_tab', 'close_tabs', 'close_browser']:
            # Close browser tabs or entire browser
            import psutil
            
            browser_lower = (browser or target or query or '').lower()
            
            # Determine if closing just tab or entire browser
            close_browser = any(word in browser_lower for word in ['browser', 'chrome', 'firefox', 'edge', 'all'])
            
            if close_browser:
                # Close entire browser
                browser_processes = []
                if 'chrome' in browser_lower:
                    browser_processes = ['chrome.exe']
                elif 'firefox' in browser_lower:
                    browser_processes = ['firefox.exe']
                elif 'edge' in browser_lower:
                    browser_processes = ['msedge.exe']
                else:
                    # Close all browsers
                    browser_processes = ['chrome.exe', 'firefox.exe', 'msedge.exe']
                
                closed = []
                for proc in psutil.process_iter(['name']):
                    try:
                        if proc.info['name'] in browser_processes:
                            proc.terminate()
                            closed.append(proc.info['name'])
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                if closed:
                    browser_names = ', '.join(set(p.replace('.exe', '').title() for p in closed))
                    return f"Successfully closed {browser_names}."
                else:
                    return "No browser windows found to close."
            else:
                # Close current tab using Ctrl+W
                try:
                    import pyautogui
                    import time
                    import subprocess
                    
                    # Try to bring browser to foreground
                    try:
                        subprocess.run(
                            ['powershell', '-Command', 
                             '(New-Object -ComObject WScript.Shell).AppActivate((Get-Process chrome,firefox,msedge -ErrorAction SilentlyContinue | Select-Object -First 1).MainWindowTitle)'],
                            capture_output=True,
                            timeout=1
                        )
                    except:
                        pass
                    
                    # Add delay for window to focus
                    time.sleep(0.5)
                    pyautogui.hotkey('ctrl', 'w')
                    time.sleep(0.1)
                    
                    return "Closed the current browser tab."
                except ImportError:
                    return "pyautogui not installed. Cannot close tabs."
                except Exception as e:
                    return f"Failed to close tab: {str(e)}"
        
        else:
            return {"error": f"Invalid action '{action}' or missing parameters. Use: open_url, youtube, gmail, search, pause, close"}
            
    except Exception as e:
        return {"error": f"Failed to open browser: {str(e)}"}

TOOL = {
    "name": "browser_control",
    "func": browser_control,
    "description": "Control browser: open URLs, play YouTube videos, open Gmail, pause/play media, close tabs/browser. Args: action='youtube'|'open_url'|'gmail'|'search'|'pause'|'close', query (video/search term OR browser name for close), url (direct URL), browser (chrome/firefox/edge). Examples: action='close' query='chrome' to close Chrome, action='close' to close current tab."
}
