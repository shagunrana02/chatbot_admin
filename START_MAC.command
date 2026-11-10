#!/bin/bash
# ChatAudit launcher for Mac - double-click this file.
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python is not installed. Download it from https://www.python.org/downloads/"
  echo "Then double-click this file again."
  read -p "Press Enter to close."
  exit 1
fi

echo "Installing required libraries (only slow the first time)..."
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install --quiet flask scikit-learn pandas

echo
echo "Starting ChatAudit. Your browser will open at http://localhost:5050"
echo "KEEP THIS WINDOW OPEN while using the app. Close it to stop."
echo
(sleep 2 && open http://localhost:5050) &
python app.py
