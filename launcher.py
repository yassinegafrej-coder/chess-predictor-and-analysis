import streamlit.web.cli as stcli
import sys
import os
import webbrowser
from threading import Timer

def resolve_path(path):
    if getattr(sys, "frozen", False):
        # If running inside an .exe
        return os.path.join(sys._MEIPASS, path)
    else:
        # If running normally
        return os.path.join(os.getcwd(), path)

def open_browser():
    webbrowser.open_new("http://localhost:8501")

if __name__ == "__main__":
    # Start a timer to open the browser 3 seconds after the app starts.
    # (3 seconds gives the app enough time to unpack and load)
    Timer(3, open_browser).start()
    
    sys.argv = [
        "streamlit", "run", 
        resolve_path("app.py"),
        "--global.developmentMode=false",
        "--server.headless=true"
    ]
    sys.exit(stcli.main())