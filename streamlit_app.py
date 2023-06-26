import streamlit as st
import sys
import subprocess

subprocess.run(["python3", "/home/appuser/app/app.py"],  stderr=sys.stdout.buffer, stdout=sys.stdout.buffer)
