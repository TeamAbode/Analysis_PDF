#!/bin/bash
# Jury Analyst Pipeline — one-click launcher (macOS)
#
# Double-click this file in Finder to start the server. A Terminal window
# will open showing logs. Close that window to stop the server.
#
# First-run setup (venv + dependencies) happens automatically.
#
# If macOS says it "cannot be opened because it is from an unidentified
# developer," right-click the file and choose Open (just once), or see
# README.md > Troubleshooting.

# Move into the script's directory regardless of where it's launched from
cd "$(dirname "$0")" || exit 1

echo "============================================================"
echo "  Jury Analyst Pipeline"
echo "============================================================"
echo ""

# --- Self-heal: strip macOS quarantine flag from the project ---
# If the project was downloaded (zip / email / browser), macOS may have
# quarantined these files. Clearing it here prevents repeated Gatekeeper
# prompts on the files this launcher touches. Harmless if nothing is flagged.
if command -v xattr >/dev/null 2>&1; then
  xattr -dr com.apple.quarantine . >/dev/null 2>&1 || true
fi

# --- Self-heal: make sure this launcher stays executable on re-runs ---
chmod +x "$0" >/dev/null 2>&1 || true

# --- Verify Python 3 is available ---
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 was not found on this Mac."
  echo ""
  echo "  Install it one of these ways, then re-run this launcher:"
  echo "    - Easiest: open Terminal and run   xcode-select --install"
  echo "    - Or download from https://www.python.org/downloads/"
  echo ""
  read -r -p "Press Return to close..." _
  exit 1
fi

# --- Create venv on first run ---
if [ ! -d ".venv" ]; then
  echo "First-run setup: creating virtual environment..."
  if ! python3 -m venv .venv; then
    echo "ERROR: failed to create the virtual environment (.venv)."
    echo "Make sure you have a full Python 3 install, then try again."
    read -r -p "Press Return to close..." _
    exit 1
  fi
  echo ""
  echo "Installing dependencies (one-time, ~30-60 seconds)..."
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --quiet --upgrade pip
  if ! pip install --quiet -r requirements.txt; then
    echo "ERROR: dependency install failed. Try running this manually:"
    echo "  cd \"$(pwd)\""
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    read -r -p "Press Return to close..." _
    exit 1
  fi
  echo "Setup complete."
  echo ""
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# --- Help Python find the Homebrew libraries (macOS) ---
# WeasyPrint loads Pango/Cairo/gdk-pixbuf at runtime via ctypes, but macOS does
# NOT search Homebrew's lib folder by default — so even when the libraries are
# installed, the import fails. Pointing DYLD_FALLBACK_LIBRARY_PATH at Homebrew's
# lib directory fixes it. (Apple Silicon: /opt/homebrew/lib, Intel: /usr/local/lib)
if command -v brew >/dev/null 2>&1; then
  BREW_LIB="$(brew --prefix)/lib"
  export DYLD_FALLBACK_LIBRARY_PATH="$BREW_LIB:${DYLD_FALLBACK_LIBRARY_PATH:-}"
fi

# --- Preflight: WeasyPrint needs native libraries (Pango/Cairo/gdk-pixbuf) ---
# These are NOT installable with pip. On a fresh Mac they're usually missing,
# which would make PDF rendering fail later with a confusing error. Catch it
# now and give clear, copy-pasteable instructions.
if ! python3 -c "import weasyprint" >/dev/null 2>&1; then
  echo "------------------------------------------------------------"
  echo "  One more setup step is needed (PDF engine libraries)."
  echo "------------------------------------------------------------"
  echo ""
  echo "  WeasyPrint needs system libraries that pip can't install:"
  echo "    pango, cairo, gdk-pixbuf, libffi"
  echo ""
  if command -v brew >/dev/null 2>&1; then
    echo "  Homebrew is installed. Installing them now..."
    if brew install pango cairo gdk-pixbuf libffi; then
      echo ""
      echo "  Done. Re-checking..."
      if ! python3 -c "import weasyprint" >/dev/null 2>&1; then
        echo "  Still not loading. Open a NEW Terminal window so the"
        echo "  library paths refresh, then re-run this launcher."
        read -r -p "Press Return to close..." _
        exit 1
      fi
    else
      echo "  brew install failed. Run it manually, then re-run this launcher:"
      echo "    brew install pango cairo gdk-pixbuf libffi"
      read -r -p "Press Return to close..." _
      exit 1
    fi
  else
    echo "  Homebrew (the installer for these) is not on this Mac."
    echo ""
    echo "  1) Install Homebrew - paste this into Terminal and follow prompts:"
    echo '       /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    echo ""
    echo "  2) Then install the libraries:"
    echo "       brew install pango cairo gdk-pixbuf libffi"
    echo ""
    echo "  3) Re-run this launcher."
    read -r -p "Press Return to close..." _
    exit 1
  fi
fi

# --- Verify .env exists and has a real key ---
if [ ! -f ".env" ]; then
  echo "NOTE: .env file not found. Creating one..."
  if [ -f ".env.example" ]; then
    cp .env.example .env
  else
    # Fallback so setup never breaks even if .env.example is missing
    printf 'ANTHROPIC_API_KEY=\n' > .env
  fi
  echo ""
  echo "  >>> You need to add your Anthropic API key to .env <<<"
  echo "  Opening it now in TextEdit. Paste your key after the = sign"
  echo "  (it looks like sk-ant-...), save with Cmd+S, then re-run this"
  echo "  launcher. Get a key at https://console.anthropic.com/"
  echo ""
  open -e .env
  read -r -p "Press Return to close..." _
  exit 0
fi

# Make sure the key is actually filled in (not blank / placeholder)
KEY_LINE="$(grep -E '^ANTHROPIC_API_KEY=' .env | head -1 | cut -d= -f2- | tr -d ' "'"'"'')"
if [ -z "$KEY_LINE" ] || [ "$KEY_LINE" = "sk-ant-..." ]; then
  echo "NOTE: your Anthropic API key isn't set yet in .env."
  echo "Opening it now - paste your key after ANTHROPIC_API_KEY= , save,"
  echo "then re-run this launcher. Get a key at https://console.anthropic.com/"
  echo ""
  open -e .env
  read -r -p "Press Return to close..." _
  exit 0
fi

# --- Free up port 8765 if something else is holding it ---
PORT=8765
if lsof -ti:$PORT >/dev/null 2>&1; then
  echo "Port $PORT is in use - freeing it up..."
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
read -r -p "Press Return to close this window..." _
