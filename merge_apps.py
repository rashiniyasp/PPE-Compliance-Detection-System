import sys

with open('app_original.py', 'r', encoding='utf-8') as f:
    orig = f.read()

with open('app.py', 'r', encoding='utf-8') as f:
    new_app = f.read()

orig_split = orig.split('st.sidebar.header("Controls")')
orig_body = 'st.sidebar.header("Controls")' + orig_split[1]

new_split = new_app.split('st.sidebar.header("Controls")')
new_body = 'st.sidebar.header("Controls")' + new_split[1]

merged = f'''import streamlit as st
import pandas as pd
import zipfile
import os
import io
import tempfile
import cv2
import time
import numpy as np
import datetime
import shutil

from pipeline import PPEPipeline

st.set_page_config(
    page_title="PPE Compliance Hub",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    :root {{
        --primary: #0ea5e9;
        --bg-dark: #0f172a;
        --card-dark: #1e293b;
    }}
    .stApp {{
        background-color: var(--bg-dark);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }}
    .metric-card {{
        background: var(--card-dark);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: bold;
        color: var(--primary);
    }}
    .metric-label {{
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .alert-text {{
        color: #ef4444;
        font-weight: bold;
    }}
</style>
""", unsafe_allow_html=True)

st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Select Module", ["Live Monitoring Dashboard", "Tracking Method Comparison"])

if app_mode == "Live Monitoring Dashboard":
    st.title("👷‍♂️ Live Monitoring Dashboard")
    st.markdown("Monitor real-time compliance for helmets, vests, gloves, goggles, and boots.")
    
{chr(10).join('    ' + line for line in orig_body.split(chr(10)))}

elif app_mode == "Tracking Method Comparison":
    st.title("⚖️ Tracking Method Comparison")
    st.markdown("Evaluate performance of SkeletonStat-Track against Baseline Trackers.")
    
{chr(10).join('    ' + line for line in new_body.split(chr(10)))}
'''

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(merged)
print("Merge complete!")
