#!/bin/bash
# Jury Analyst Pipeline — one-click launcher (macOS)
#
# Double-click this file in Finder to start the server. A Terminal window
# will open showing logs. Close that window to stop the server.
#
# First-run setup (venv + dependencies) happens automatically.

# Move into the script's directory regardless of where it's launched from
cd "$(dirname "$0")"

echo "============================================================"
echo "  Jury Analyst Pipeline"
echo "============================================================"
echo ""

# --- Verify Python 3 is available ---
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found."
  echo "Install Python 3 from https://www.python.org/downloads/ and try again."
  echo ""
  read -p "Press Return to close..." _
  exit 1
fi

# --- Create venv on first run ---
if [ ! -d ".venv" ]; then
  echo "First-run setup: creating virtual environment..."
  python3 -m venv .venv
  if [ $? -ne 0 ]; then
    echo "ERROR: failed to create venv."
    read -p "Press Return to close..." _
    exit 1
  fi
  echo ""
  echo "Installing dependencies (one-time, ~30 seconds)..."
  source .venv/bin/activate
  pip install --quiet --upgrade pip
  pip install --quiet -r requirements.txt
  if [ $? -ne 0 ]; then
    echo "ERROR: dependency install failed. Try running this manually:"
    echo "  cd \"$(pwd)\""
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    read -p "Press Return to close..." _
    exit 1
  fi
  echo "Setup complete."
  echo ""
else
  source .venv/bin/activate
fi

# --- Verify .env exists ---
if [ ! -f ".env" ]; then
  echo "NOTE: .env file not found. Creating from example..."
  cp .env.example .env
  echo ""
  echo "  >>> You need to add your Anthropic API key to .env <<<"
  echo "  Opening it now in TextEdit. Paste your key after the = sign,"
  echo "  save with Cmd+S, then re-run this launcher."
  echo ""
  open -e .env
  read -p "Press Return to close..." _
  exit 0
fi

# --- Free up port 8765 if something else is holding it ---
PORT=8765
if lsof -ti:$PORT >/dev/null 2>&1; then
  echo "Port $PORT is in use — freeing it up..."
  lsof -ti:$PORT | xargs kill -9 2>/dev/null
  sleep 1
fi

# --- Open browser in 3 seconds (in parallel with server startup) ---
( sleep 3 && open "http://localhost:$PORT" ) &

# --- Start the server (foreground; closing Terminal stops it) ---
echo "Starting server on http://localhost:$PORT"
echo "Your browser will open automatically in a moment."
echo ""
echo "To stop the server: close this Terminal window."
echo "============================================================"
echo ""

python3 -u app.py

# If app.py exits cleanly, hold the window so you can read any errors
echo ""
echo "Server stopped."
read -p "Press Return to close this window..." _
