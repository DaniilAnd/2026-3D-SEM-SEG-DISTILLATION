#!/bin/bash
# Run the Streamlit 3D Point Cloud Patching Visualization app

cd "$(dirname "$0")"
streamlit run app/main.py --server.port 8501
