# Jury Analyst Pipeline — Setup Guide

This guide walks you through installing and running the Jury Analyst Pipeline on
your own computer. Follow the section for your operating system (**Mac** or
**Windows**). You'll copy-paste a few commands into a terminal — no coding
required.

**What you'll end up with:** the app running in your web browser at
`http://localhost:8765`, where you do the 3-phase analysis (Clean → Analyze →
Report) and download the final PDF.

**Time:** about 10–15 minutes the first time. After that, starting it takes
seconds.

**You'll need:** an Anthropic API key (looks like `sk-ant-...`). Ask your
administrator for the firm's key, or create your own at
https://console.anthropic.com/ under **API Keys**.

---

# 🍎 Mac instructions

## Step 1 — Install the prerequisites (one time)

Open the **Terminal** app: press `Cmd + Space`, type `Terminal`, press Return.

**a) Install Homebrew** (a free installer for the libraries the app needs).
Copy-paste this line, press Return, and follow the prompts (it may ask for your
Mac password — typing won't show characters, that's normal):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

When it finishes, it may print 1–2 lines starting with `eval` and ask you to run
them so the `brew` command is recognized. If so, copy-paste those lines and
press Return. (If you're not sure, just **close Terminal and open a new one**.)

**b) Install Python and Git.** Paste this and press Return:

```bash
brew install python git
```

> Already have these? Running the command again is harmless — it just confirms
> they're up to date.

## Step 2 — Download the app (one time)

In Terminal, paste these lines (this puts the app in a folder on your Desktop):

```bash
cd ~/Desktop
git clone https://github.com/TeamAbode/Analysis_PDF.git
cd Analysis_PDF
```

## Step 3 — Add your API key (one time)

Paste the line below, **but first replace `sk-ant-PASTE-YOUR-KEY-HERE` with your
real key** (keep the single quotes):

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-PASTE-YOUR-KEY-HERE' > .env
cat .env
```

The `cat .env` line prints the file back so you can confirm your full key is
there after the `=`.

## Step 4 — Start the app

Paste:

```bash
./start.command
```

The **first** run takes a minute or two — it sets up a private environment,
installs the app's components, and installs the PDF libraries automatically.
You'll see lots of text scroll by; that's normal. When it's ready, your browser
opens to the app.

> **If the browser shows "This site can't be reached":** the app is still
> starting up. Wait about 15 seconds, then refresh the page (`Cmd + R`).

That's it — do your analysis in the browser. ✅

## Starting it again next time

You don't repeat the setup. Just:

```bash
cd ~/Desktop/Analysis_PDF
./start.command
```

(Or open the `Analysis_PDF` folder in Finder and double-click `start.command`.)

**To stop the app:** close the Terminal window that's running it.

---

# 🪟 Windows instructions

## Step 1 — Install the prerequisites (one time)

**a) Install Python.** Download it from https://www.python.org/downloads/ and
run the installer. **IMPORTANT:** on the first screen, tick the box that says
**"Add python.exe to PATH"** before clicking Install. (If you forget this,
nothing will work.)

**b) Install Git.** Download from https://git-scm.com/download/win and run the
installer (the default options are fine — just keep clicking Next).

**c) Install the PDF engine libraries (GTK3).** The PDF generator needs these.
Download the installer from:
https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases
(get the latest `*.exe`), run it, and keep the default options.

**After installing all three, restart your computer** so everything is
recognized.

## Step 2 — Download the app (one time)

Open **Command Prompt**: press the Windows key, type `cmd`, press Enter. Then
paste these lines (puts the app on your Desktop):

```bat
cd %USERPROFILE%\Desktop
git clone https://github.com/TeamAbode/Analysis_PDF.git
cd Analysis_PDF
```

## Step 3 — Add your API key (one time)

Paste the line below, **but first replace `sk-ant-PASTE-YOUR-KEY-HERE` with your
real key**:

```bat
echo ANTHROPIC_API_KEY=sk-ant-PASTE-YOUR-KEY-HERE> .env
type .env
```

(The `type .env` line prints it back so you can check the key is there. Make
sure there's no space before the `>`.)

## Step 4 — Start the app

In the same window, paste:

```bat
start.bat
```

(Or open the `Analysis_PDF` folder in File Explorer and double-click
`start.bat`.) The first run takes a minute or two to set everything up, then
your browser opens to the app.

> **If the browser shows the page can't be reached:** wait ~15 seconds and
> refresh (`Ctrl + R`).

## Starting it again next time

```bat
cd %USERPROFILE%\Desktop\Analysis_PDF
start.bat
```

(Or double-click `start.bat` in the folder.) **To stop:** close the command
window.

---

# Using the app

Once it's open in your browser at `http://localhost:8765`:

1. **Phase 1 — Clean:** upload the Alchemer CSV + survey PDF, review/adjust the
   filters, and commit the clean dataset.
2. **Phase 2 — Analyze:** confirm the case details, run the analysis (charts &
   stats are generated automatically).
3. **Phase 3 — Report:** the AI writes the narrative sections. Edit any section
   you want, then **download the PDF**.

Your files are saved per-case inside the app's `workspace` folder on your
computer.

---

# Troubleshooting

**`zsh: no such file or directory: ./start.command` (Mac)** — you're not inside
the app folder. Run `cd ~/Desktop/Analysis_PDF` first, then `./start.command`.
Tip: when the Terminal prompt ends in `Analysis_PDF %` you're in the right
place; when it ends in `~ %` you're in your home folder.

**"start.command can't be opened because it is from an unidentified developer"
(Mac)** — only happens if you downloaded a ZIP instead of using `git clone`.
Right-click `start.command` → **Open** → **Open** (one time only).

**"could not be executed because you do not have appropriate access privileges"
(Mac)** — run this once in the app folder, then try again:
```bash
chmod +x start.command
```

**"The document '.env' could not be saved. The file is locked" (Mac)** — ignore
TextEdit; set the key from Terminal instead using the Step 3 command above.

**The PDF step / WeasyPrint error (Mac)** — make sure Homebrew is installed
(Step 1a). The launcher installs the rest automatically. If it still complains,
**open a brand-new Terminal window** and run `./start.command` again.

**"python3 was not found" / "Python was not found"** — Python isn't installed
(or, on Windows, "Add to PATH" wasn't ticked). Reinstall Python per Step 1.

**"ANTHROPIC_API_KEY not set" or the AI sections fail** — your key isn't in the
`.env` file. Redo Step 3, making sure your real `sk-ant-...` key follows the
`=`.

**The browser says the site can't be reached** — the app just needs a few more
seconds to start. Wait ~15 seconds and refresh. Make sure the Terminal/command
window running the app is still open (closing it stops the app).

---

# Getting an Anthropic API key

1. Go to https://console.anthropic.com/ and sign in (or create an account).
2. Open **API Keys** → **Create Key**.
3. Copy the key (starts with `sk-ant-`) and use it in Step 3.

> Keep your key private — anyone with it can use your account's credit. If the
> firm provides a shared key, use that one.
