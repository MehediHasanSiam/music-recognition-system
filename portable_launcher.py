import os
import sys
import threading
import webbrowser
from pathlib import Path

# Import project modules so PyInstaller includes them
import audio_utils
import database
import features
import fingerprint
import clip_matcher
import hum_matcher
import recommender

from streamlit.web import cli as stcli


def get_app_root():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def main():
    app_root = get_app_root()

    # Make relative paths like data/music.db work
    os.chdir(app_root)

    # app.py is bundled inside the PyInstaller package
    if getattr(sys, "frozen", False):
        bundled_root = Path(sys._MEIPASS)
    else:
        bundled_root = Path(__file__).resolve().parent

    app_script = bundled_root / "app.py"

    url = "http://127.0.0.1:8501"

    # Open the browser shortly after Streamlit starts
    threading.Timer(
        2.0,
        lambda: webbrowser.open(url)
    ).start()

    sys.argv = [
    "streamlit",
    "run",
    str(app_script),
    "--global.developmentMode=false",
    "--server.address=127.0.0.1",
    "--server.port=8501",
    "--server.headless=true",
    "--browser.gatherUsageStats=false",
 ]

    sys.exit(stcli.main())


if __name__ == "__main__":
    main()