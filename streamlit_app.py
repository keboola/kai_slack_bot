import streamlit as st
import sys
import subprocess

subprocess.run(["python3", "app.py"],  stderr=sys.stdout.buffer, stdout=sys.stdout.buffer)
