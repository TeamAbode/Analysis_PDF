# Hosting the app so anyone can use it (no install)

This guide puts the Jury Analyst Pipeline online at a web address. Anyone you
give the link to can open it in a browser and use it — no Python, no Homebrew,
no setup on their machine. All AI runs are billed to the **one** Anthropic API
key you set on the server.

The app is packaged as a Docker container (it bundles the PDF libraries), so it
runs the same everywhere. The steps below use **Render**, which is the simplest
host for this. (Railway or Fly.io work too; the Dockerfile is standard.)

---

## What you need first

1. **An Anthropic API key** (the firm's) — from https://console.anthropic.com/.
   This is the key that pays for everyone's runs.
2. **A Render account** — sign up free at https://render.com (you can log in
   with GitHub).
3. This repo on GitHub (it already is: `TeamAbode/Analysis_PDF`).

---

## Deploy in 6 steps

1. Log in to **https://dashboard.render.com**.
2. Click **New +** (top right) → **Blueprint**.
3. **Connect** the `TeamAbode/Analysis_PDF` repository when prompted, and pick
   the branch `claude/report-generation-interface-s7pfon` (or `main` once this
   is merged). Render finds the `render.yaml` file automatically.
4. Render shows the service it will create (`jury-analyst`) and asks for the
   secret values:
   - **ANTHROPIC_API_KEY** → paste the firm's key (`sk-ant-...`).
   - **APP_PASSWORD** → leave **blank** for an open link (anyone with the URL
     can use it). See "Locking it later" below.
5. Click **Apply** / **Create**. The first build takes a few minutes (it's
   building the container). You'll see logs; wait for **"Live"**.
6. Render gives you a URL like **`https://jury-analyst.onrender.com`**. That's
   the link — open it, and share it with anyone who needs it.

That's it. To use the app, people just open that link.

---

## Locking it later (optional, recommended eventually)

Right now anyone with the link can use it, and every run spends the firm's API
budget. You can require a password at any time — **no code change**:

1. Render dashboard → your service → **Environment**.
2. Set **APP_PASSWORD** to a password of your choice → **Save**.
3. The app restarts automatically. Now the browser asks for a password the
   first time someone visits (username can be anything). Share the password
   only with people who should have access.

To go open again, clear that variable and save.

---

## Updating the app later

Because `autoDeploy` is on, any push to the connected branch makes Render
rebuild and redeploy automatically. Just commit and push your changes — the
live site updates in a few minutes.

---

## Two things to know

- **Generated case files don't persist across redeploys.** The server's disk is
  temporary, so a case you create lives only until the next deploy/restart. In
  practice that's fine: you run a case and download the PDF in the same session.
  If you later want cases to survive restarts, add a Render **Disk** mounted at
  `/app/workspace` (a paid add-on) — tell me and I'll wire it into `render.yaml`.
- **Cost.** You pay Render for the always-on service (the `starter` plan is a
  few dollars a month; the `free` plan works but sleeps when idle and is slow to
  wake). Anthropic usage is billed separately to your API key, per report.
