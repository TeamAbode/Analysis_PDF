// Jury Analyst — frontend helpers
const API = "/api";

function getCaseId() {
  // Pull from URL ?case=XXX or fall back to localStorage
  const url = new URL(window.location.href);
  let caseId = url.searchParams.get("case");
  if (!caseId) {
    caseId = localStorage.getItem("ja_case_id") || "";
  } else {
    localStorage.setItem("ja_case_id", caseId);
  }
  return caseId;
}

function setCaseId(caseId) {
  localStorage.setItem("ja_case_id", caseId);
}

function showNotice(msg, kind = "") {
  const el = document.getElementById("notice");
  if (!el) return;
  el.className = "notice " + kind;
  el.textContent = msg;
  el.style.display = "block";
}

function hideNotice() {
  const el = document.getElementById("notice");
  if (el) el.style.display = "none";
}

async function api(method, path, body, isForm = false) {
  const opts = { method };
  if (body) {
    if (isForm) {
      opts.body = body;
    } else {
      opts.headers = { "Content-Type": "application/json" };
      opts.body = JSON.stringify(body);
    }
  }
  const r = await fetch(API + path, opts);
  if (!r.ok) {
    let err;
    try { err = await r.json(); } catch { err = { detail: r.statusText }; }
    throw new Error(err.detail || `HTTP ${r.status}`);
  }
  const ct = r.headers.get("content-type") || "";
  if (ct.includes("application/json")) return r.json();
  return r.blob();
}

function fmt$(v) {
  if (v == null) return "—";
  return "$" + Number(v).toLocaleString();
}

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "className") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === "html") e.innerHTML = v;
    else e.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    if (typeof c === "string") e.appendChild(document.createTextNode(c));
    else e.appendChild(c);
  }
  return e;
}

function renderCrumbs(active) {
  const c = document.querySelector(".brand-bar .crumbs");
  if (!c) return;
  const caseId = getCaseId();
  const cq = caseId ? `?case=${encodeURIComponent(caseId)}` : "";
  const items = [
    { label: "Start", href: "/", key: "start" },
    { label: "1. Clean", href: "/phase1.html" + cq, key: "1" },
    { label: "2. Analyze", href: "/phase2.html" + cq, key: "2" },
    { label: "3. Report", href: "/phase3.html" + cq, key: "3" },
  ];
  c.innerHTML = "";
  items.forEach((it, i) => {
    const s = document.createElement("span");
    if (it.key === active) s.className = "active";
    const a = document.createElement("a");
    a.href = it.href;
    a.textContent = it.label;
    a.style.color = "inherit";
    s.appendChild(a);
    c.appendChild(s);
    if (i < items.length - 1) c.appendChild(document.createTextNode(" › "));
  });
}
