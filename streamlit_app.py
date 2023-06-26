import streamlit as st

import subprocess

subprocess.run(["python3", "app.py"], capture_output=True, stdout=subprocess.STDOUT, stderr=subprocess.STDOUT)
