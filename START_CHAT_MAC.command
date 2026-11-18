#!/bin/bash
# ChatAudit full chatbot app for Mac - double-click this file.
cd "$(dirname "$0")"
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install --quiet flask scikit-learn pandas
echo "Starting the ChatAudit app at http://localhost:5051 - keep this window open."
(sleep 2 && open http://localhost:5051) &
python chat_app.py
