import os
import subprocess
import streamlit
from Analyse_und_Machine_Learning import zielverzeichnis

script_path = os.path.join(zielverzeichnis, "streamlit_app.py")

subprocess.run(['streamlit', 'run', script_path])
