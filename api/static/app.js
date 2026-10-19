// GraphRAG console (ROADMAP D5). Plain JavaScript, no build step and no network dependency.
// Every string from the corpus or the model is inserted as text, never as HTML.
"use strict";

// Storage can be refused (private windows, embedded frames); the console must work without it.
const store = {
  get(key) { try { return sessionStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { sessionStorage.setItem(key, value); } catch { /* kept in memory only */ } },
};

const state = {
  token: store.get("graphrag-token") || "",
  session: null,
  docs: { offset: 0, limit: 50, total: 0, search: "", status: "" },
  loaded: { corpus: false, graph: false },
  jobsTimer: null,
  knownJobs: new Map(),
};

// ---------------------------------------------------------------- helpers

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

const $ = (id) => document.getElementById(id);
const fmt = (n) => (typeof n === "number" ? n.toLocaleString() : n ?? "—");

function when(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}

function duration(ms) {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const response = await fetch(path, { ...options, headers });
  if (response.status === 401) {
    if (await askToken()) return api(path, options);
    throw new Error("This API needs a token.");
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body.detail;
    throw new Error(typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : `HTTP ${response.status}`);
  }
  return body;
}

function askToken() {
  return new Promise((resolve) => {
    const dialog = $("token-dialog");
    $("token-input").value = "";
    dialog.onclose = () => {
      const value = $("token-input").value.trim();
      if (dialog.returnValue === "ok" && value) {
        state.token = value;
        store.set("graphrag-token", value);
        resolve(true);
      } else resolve(false);
    };
    dialog.showModal();
  });
}

function confirmAction(title, text, okLabel = "Remove") {
  return new Promise((resolve) => {
    const dialog = $("confirm-dialog");
    $("confirm-title").textContent = title;
    $("confirm-text").textContent = text;
    dialog.querySelector('button[value="ok"]').textContent = okLabel;
    dialog.onclose = () => resolve(dialog.returnValue === "ok");
    dialog.showModal();
  });
}

// ---------------------------------------------------------------- tabs and health

function showTab(name) {
  if (location.hash.slice(1) !== name) history.replaceState(null, "", `${location.pathname}${location.search}#${name}`);
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  for (const tab of ["ask", "corpus", "graph"]) $(`tab-${tab}`).hidden = tab !== name;
  if (name === "corpus" && !state.loaded.corpus) { state.loaded.corpus = true; loadDocs(); loadJobs(); }
  if (name === "graph" && !state.loaded.graph) { state.loaded.graph = true; loadOverview(); }
  if (name === "ask") $("question").focus();
}

async function refreshHealth() {
  const pill = $("status");
  try {
    const health = await api("/health");
    const docs = health.stores?.sqlite?.complete ?? 0;
    pill.textContent = `${health.status === "ok" ? "Healthy" : "Degraded"} · ${fmt(docs)} documents`;
    pill.className = `status-pill ${health.status}`;
    renderHealth(health);
  } catch (error) {
    pill.textContent = "Unreachable";
    pill.className = "status-pill degraded";
  }
}

function renderHealth(health) {
  const box = $("health");
  const names = { sqlite: "SQLite", neo4j: "Neo4j", qdrant: "Qdrant", redis: "Redis" };
  const chips = Object.entries(health.stores || {}).map(([key, value]) => {
    const note = value.fallback ? " (fallback)" : value.ok ? "" : ` (${value.error || "down"})`;
    return h("span", { class: "chip", title: JSON.stringify(value) }, h("span", { class: `dot ${value.ok ? "" : "bad"}` }), `${names[key] || key}${note}`);
  });
  const c = health.communities || {};
  const community = c.recorded
    ? `Communities: ${fmt(c.changed_documents)} of ${fmt(c.threshold)} changed documents before a rebuild`
    : "Communities: no build recorded yet";
  box.replaceChildren(...chips, h("span", { class: "muted small" }, community),
    h("span", { class: "muted small" }, `Models: ${health.models?.generation} · ${health.models?.embedding} · ${health.models?.rerank || "no rerank"}`));
}

// ---------------------------------------------------------------- ask

const WARNINGS = {
  unsupported_claims_removed: "Some unsupported claims were removed",
  claims_rest_on_flagged_evidence: "Rests on evidence flagged for review",
  corpus_update_in_progress: "Corpus update in progress",
  invalid_or_unsupported_claim_evidence_references: "Some claims failed validation",
};
const REFUSALS = new Set(["insufficient_evidence", "insufficient_validated_evidence"]);

function renderAnswer(reply) {
  const refused = reply.warnings.some((w) => REFUSALS.has(w)) || reply.claims.length === 0;
  const numbers = new Map(reply.citations.map((c, i) => [c.citation_id, i + 1]));
  const head = h("div", { class: "answer-head" });
  if (reply.verdict && !refused) {
    const kind = reply.verdict === "yes" ? "yes" : reply.verdict === "no" ? "no" : "";
    head.append(h("span", { class: `badge ${kind}` }, reply.verdict));
  }
  if (refused) head.append(h("span", { class: "badge warn" }, "Not answered from the corpus"));
  head.append(h("span", { class: "badge" }, reply.query_type));
  if (reply.cached) head.append(h("span", { class: "badge" }, "cached"));
  for (const warning of reply.warnings) {
    if (REFUSALS.has(warning)) continue;
    head.append(h("span", { class: "badge warn", title: warning }, WARNINGS[warning] || warning.replaceAll("_", " ")));
  }

  const sources = h("ol", { class: "sources" }, reply.citations.map((citation, index) => {
    const doc = reply.documents?.[citation.doc_id] || {};
    const title = doc.title || citation.doc_name || citation.doc_id || citation.chunk_id;
    const meta = [doc.publisher, doc.published, citation.page ? `p. ${citation.page}` : null].filter(Boolean).join(" · ");
    return h("li", { class: "source" }, h("details", {},
      h("summary", {}, h("span", { class: "num" }, `[${index + 1}]`), h("span", {}, title, meta ? h("span", { class: "meta" }, ` — ${meta}`) : null)),
      h("div", { class: "excerpt" }, citation.excerpt || "(no excerpt)"),
    ));
  }));

  const claims = reply.claims.length
    ? h("details", { class: "more" }, h("summary", {}, `Claims (${reply.claims.length})`),
        h("ol", {}, reply.claims.map((claim) => {
          const refs = claim.citation_ids.map((id) => numbers.get(id)).filter(Boolean).map((n) => `[${n}]`).join("");
          return h("li", {}, claim.text, refs ? h("span", { class: "muted" }, ` ${refs}`) : null);
        })))
    : null;
  const paths = reply.graph_paths_used.length
    ? h("details", { class: "more" }, h("summary", {}, `Graph paths (${reply.graph_paths_used.length})`),
        h("ul", { class: "paths" }, reply.graph_paths_used.map((p) => h("li", {}, p))))
    : null;
  const total = Object.values(reply.stage_timings_ms || {}).reduce((a, b) => a + b, 0);
  const foot = h("div", { class: "answer-foot" },
    reply.cached ? `From the cache in ${duration(total)}` : `Answered in ${duration(total)}`,
    reply.claims.length ? ` · support ${(reply.confidence * 100).toFixed(0)}%` : "");

  return h("article", { class: "answer" }, head,
    reply.interpreted_query ? h("div", { class: "read-as" }, `Read as: ${reply.interpreted_query}`) : null,
    h("div", { class: `answer-text ${refused ? "refused" : ""}` }, reply.answer),
    reply.citations.length ? sources : null, claims, paths, foot);
}

async function ask(question) {
  question = question.trim();
  if (!question) return;
  $("empty").hidden = true;
  const thread = $("thread");
  thread.append(h("div", { class: "question" }, question));
  const pending = h("div", { class: "thinking" }, "Retrieving and checking evidence");
  thread.append(pending);
  pending.scrollIntoView({ behavior: "smooth", block: "end" });
  const button = $("ask");
  button.disabled = true;
  try {
    const reply = await api("/query", {
      method: "POST",
      body: JSON.stringify({ question, session_id: state.session }),
    });
    state.session = reply.session_id;
    pending.replaceWith(renderAnswer(reply));
    $("session-line").textContent = `Conversation ${reply.session_id} · follow-ups are read in the light of earlier turns`;
  } catch (error) {
    pending.replaceWith(h("article", { class: "answer" }, h("span", { class: "badge warn" }, "Error"), " ", error.message));
  } finally {
    button.disabled = false;
    thread.lastElementChild?.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function newConversation() {
  state.session = null;
  $("thread").replaceChildren();
  $("empty").hidden = false;
  $("session-line").textContent = "";
  $("question").focus();
}

async function loadSuggestions() {
  const box = $("suggestions");
  let prompts = [];
  try {
    const overview = await api("/graph/overview");
    prompts = (overview.communities || []).slice(0, 4).map((c) => `What does the corpus say about ${c.title.charAt(0).toLowerCase()}${c.title.slice(1)}?`);
  } catch { /* the suggestions are a convenience */ }
  if (!prompts.length) prompts = ["What are the broad subject areas this corpus covers?"];
  box.replaceChildren(...prompts.map((p) => h("button", { type: "button", onclick: () => ask(p) }, p)));
}

// ---------------------------------------------------------------- corpus

async function loadDocs() {
  const { offset, limit, search, status } = state.docs;
  const params = new URLSearchParams({ offset, limit });
  if (search) params.set("search", search);
  if (status) params.set("status", status);
  const rows = $("doc-rows");
  try {
    const page = await api(`/documents?${params}`);
    state.docs.total = page.total;
    rows.replaceChildren(...page.documents.map((doc) => h("tr", { onclick: () => openDoc(doc.doc_id) },
      h("td", { class: "title" }, doc.title || doc.source_path),
      h("td", {}, doc.publisher || ""),
      h("td", { class: "nowrap" }, doc.published || ""),
      h("td", { class: "nowrap" }, doc.format || ""),
      h("td", { class: "nowrap" }, h("span", { class: `status-dot ${doc.status}` }), doc.status || ""),
      h("td", { class: "num" }, fmt(doc.active_chunks)))));
    if (!page.documents.length) rows.replaceChildren(h("tr", {}, h("td", { colspan: 6, class: "muted" }, "No documents match.")));
    const end = Math.min(offset + limit, page.total);
    $("doc-range").textContent = page.total ? `${fmt(offset + 1)}–${fmt(end)} of ${fmt(page.total)}` : "";
    $("doc-prev").disabled = offset === 0;
    $("doc-next").disabled = end >= page.total;
  } catch (error) {
    rows.replaceChildren(h("tr", {}, h("td", { colspan: 6 }, `Could not load documents: ${error.message}`)));
  }
}

async function openDoc(docId) {
  const box = $("doc-detail");
  box.replaceChildren(h("p", { class: "muted" }, "Loading…"));
  $("doc-dialog").showModal();
  try {
    const doc = await api(`/documents/${encodeURIComponent(docId)}`);
    const remove = doc.active_version_id
      ? h("button", { class: "danger", onclick: () => removeDoc(doc) }, "Remove from corpus")
      : h("p", { class: "muted small" }, "Not in the corpus: it has no active version.");
    box.replaceChildren(
      h("h2", {}, doc.title || doc.doc_id),
      h("dl", { class: "kv" },
        h("dt", {}, "Publisher"), h("dd", {}, doc.publisher || "—"),
        h("dt", {}, "Published"), h("dd", {}, doc.published || "—"),
        h("dt", {}, "Format"), h("dd", {}, doc.format || "—"),
        h("dt", {}, "Status"), h("dd", {}, doc.status || "—"),
        h("dt", {}, "Active chunks"), h("dd", {}, fmt(doc.active_chunks)),
        h("dt", {}, "Source"), h("dd", { class: "detail-path" }, doc.source_path, doc.source_exists ? "" : " (file missing)"),
        h("dt", {}, "Document id"), h("dd", { class: "detail-path" }, doc.doc_id)),
      h("h2", { style: "margin-top:1rem" }, "Versions"),
      h("ul", { class: "versions" }, doc.versions.map((v) => h("li", {}, `${v.status} — ${when(v.created_at)} — `, h("code", {}, v.version_id.slice(0, 8))))),
      h("div", { style: "margin-top:1rem" }, remove));
  } catch (error) {
    box.replaceChildren(h("p", {}, `Could not load the document: ${error.message}`));
  }
}

async function removeDoc(doc) {
  const ok = await confirmAction("Remove this document?",
    `“${doc.title || doc.doc_id}” stops being used in answers at once; its assertions and vectors are then retracted. Entities other documents support are kept. Re-ingesting the file brings it back.`);
  if (!ok) return;
  try {
    await api(`/documents/${encodeURIComponent(doc.doc_id)}`, { method: "DELETE" });
    $("doc-dialog").close();
    loadJobs();
  } catch (error) {
    alert(`Removal was refused: ${error.message}`);
  }
}

async function submitIngest(event) {
  event.preventDefault();
  try {
    await api("/ingest", {
      method: "POST",
      body: JSON.stringify({ path: $("ingest-path").value.trim(), recursive: $("ingest-recursive").checked }),
    });
    loadJobs();
  } catch (error) {
    alert(`Ingest was refused: ${error.message}`);
  }
}

async function runJob(kind) {
  if (kind === "communities") {
    const ok = await confirmAction("Rebuild the communities now?",
      "Each community is summarised by the model, so this makes roughly one provider call per community, and replaces the build that GLOBAL questions use.", "Rebuild");
    if (!ok) return;
  }
  try {
    await api(`/jobs/${kind}`, { method: "POST" });
    loadJobs();
  } catch (error) {
    alert(`The job was refused: ${error.message}`);
  }
}

function jobSummary(job) {
  if (job.error) return job.error;
  if (!job.result) return job.params ? JSON.stringify(job.params) : "";
  const r = job.result;
  if (job.kind === "ingest") return `processed ${r.processed}, changed ${r.changed}, skipped ${r.skipped}, failed ${r.failed}`;
  if (job.kind === "remove") return `retraction ${r.retraction}; left in Neo4j ${JSON.stringify(r.left_in_neo4j)}`;
  if (job.kind === "communities") return `summarised ${r.summarised} of ${r.kept} communities (${r.failed} failed)`;
  if (job.kind === "heal-report") return `orphans ${r.orphans_found}, functional conflicts ${r.functional_conflicts}, type variance ${r.type_conflicts}`;
  return JSON.stringify(r);
}

async function loadJobs() {
  clearTimeout(state.jobsTimer);
  const list = $("jobs");
  try {
    const { jobs } = await api("/jobs?limit=20");
    let finishedNow = false;
    for (const job of jobs) {
      const before = state.knownJobs.get(job.job_id);
      if (before && before !== job.status && (job.status === "succeeded" || job.status === "failed")) finishedNow = true;
      state.knownJobs.set(job.job_id, job.status);
    }
    list.replaceChildren(...(jobs.length ? jobs.map((job) => h("li", { class: "job" },
      h("div", { class: "job-head" }, h("strong", {}, job.kind), h("span", { class: `badge ${job.status}` }, job.status)),
      h("div", { class: "muted small" }, `${when(job.created_at)}${job.params?.path ? ` · ${job.params.path}` : ""}${job.params?.doc_id ? ` · ${job.params.doc_id.slice(0, 8)}` : ""}`),
      h("pre", {}, jobSummary(job)))) : [h("li", { class: "muted small" }, "No jobs yet.")]));
    if (finishedNow) { loadDocs(); refreshHealth(); state.loaded.graph = false; }
    if (jobs.some((j) => j.status === "queued" || j.status === "running")) state.jobsTimer = setTimeout(loadJobs, 3000);
  } catch (error) {
    list.replaceChildren(h("li", { class: "muted small" }, `Could not load jobs: ${error.message}`));
  }
}

// ---------------------------------------------------------------- graph

function bars(container, rows, { name = "name", value = "count", note } = {}) {
  const max = Math.max(1, ...rows.map((r) => r[value]));
  container.replaceChildren(...rows.map((row) => h("div", { class: "bar-row", title: `${row[name]}: ${fmt(row[value])}${note ? ` · ${note(row)}` : ""}` },
    h("span", { class: "name" }, row[name] || "—"),
    h("span", { class: "bar-track" }, h("span", { class: "bar-fill", style: `width:${(100 * row[value]) / max}%` })),
    h("span", { class: "count" }, fmt(row[value]), note ? h("span", { class: "muted" }, ` · ${note(row)}`) : null))));
  if (!rows.length) container.replaceChildren(h("p", { class: "muted small" }, "Nothing to show."));
}

function densityChart(container, series) {
  if (!series.length) { container.replaceChildren(h("p", { class: "muted small" }, "No activations recorded.")); return; }
  const width = 760, height = 220, pad = { l: 46, r: 46, t: 12, b: 28 };
  const xs = (i) => pad.l + (series.length === 1 ? (width - pad.l - pad.r) / 2 : (i * (width - pad.l - pad.r)) / (series.length - 1));
  const maxDocs = Math.max(1, ...series.map((s) => s.documents));
  const maxDensity = Math.max(1, ...series.map((s) => s.assertions_per_document));
  const yDocs = (v) => height - pad.b - (v / maxDocs) * (height - pad.t - pad.b);
  const yDensity = (v) => height - pad.b - (v / maxDensity) * (height - pad.t - pad.b);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("class", "chart");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Documents and assertions per document over time");
  const add = (tag, attrs, text) => {
    const el = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    if (text !== undefined) el.textContent = text;
    svg.append(el);
  };
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (i * (height - pad.t - pad.b)) / 4;
    add("line", { x1: pad.l, x2: width - pad.r, y1: y, y2: y, class: "grid" });
    add("text", { x: pad.l - 6, y: y + 4, "text-anchor": "end" }, fmt(Math.round(maxDocs * (1 - i / 4))));
    add("text", { x: width - pad.r + 6, y: y + 4 }, (maxDensity * (1 - i / 4)).toFixed(1));
  }
  const path = (y, key) => series.map((s, i) => `${i ? "L" : "M"}${xs(i).toFixed(1)},${y(s[key]).toFixed(1)}`).join(" ");
  add("path", { d: path(yDocs, "documents"), class: "line-docs" });
  add("path", { d: path(yDensity, "assertions_per_document"), class: "line-density" });
  const step = Math.max(1, Math.ceil(series.length / 8));
  series.forEach((s, i) => { if (i % step === 0 || i === series.length - 1) add("text", { x: xs(i), y: height - 8, "text-anchor": "middle" }, s.day.slice(5)); });
  container.replaceChildren(svg, h("div", { class: "legend" },
    h("span", {}, h("i", { style: "background:var(--accent)" }), "Documents (left axis)"),
    h("span", {}, h("i", { style: "background:var(--yes)" }), "Assertions per document (right axis)")));
}

async function loadOverview() {
  const tiles = $("graph-tiles");
  tiles.replaceChildren(h("p", { class: "muted" }, "Reading the graph…"));
  try {
    const o = await api("/graph/overview");
    const g = o.graph || {};
    const tile = (value, label) => h("div", { class: "tile" }, h("div", { class: "value" }, fmt(value)), h("div", { class: "label" }, label));
    const accepted = g.assertions ? `${Math.round((100 * g.accepted_assertions) / g.assertions)}%` : "—";
    tiles.replaceChildren(tile(o.documents, "Documents"), tile(g.entities, "Entities"), tile(g.assertions, "Assertions"),
      tile(accepted, "Assertions accepted"), tile((o.communities || []).length, "Communities"));
    if (Object.keys(o.errors || {}).length) tiles.append(h("p", { class: "muted small" }, `Unavailable: ${Object.entries(o.errors).map(([k, v]) => `${k} (${v})`).join(", ")}`));
    bars($("entity-types"), g.entity_types || []);
    bars($("relation-types"), g.relation_types || []);
    densityChart($("density"), o.density || []);
    bars($("communities"), (o.communities || []).slice(0, 60), { name: "title", value: "size", note: (r) => `${r.documents} docs` });
    const b = o.community_build || {};
    $("community-build").textContent = b.recorded
      ? `Build ${b.build_id === "adopted" ? "adopted as baseline" : b.build_id?.slice(0, 8)} · ${when(b.built_at)} · ${fmt(b.changed_documents)} of ${fmt(b.threshold)} changed documents before a rebuild`
      : "No community build recorded yet.";
    const heal = o.healer;
    $("healer").replaceChildren(heal && heal.available
      ? h("dl", { class: "kv" },
          h("dt", {}, "Run"), h("dd", {}, when(heal.at)),
          h("dt", {}, "Entities with no assertions"), h("dd", {}, fmt(heal.orphans_found)),
          h("dt", {}, "Type disagreements"), h("dd", {}, fmt(heal.type_conflicts)),
          h("dt", {}, "Functional conflicts"), h("dd", {}, fmt(heal.functional_conflicts)),
          h("dt", {}, "Low-confidence, uncorroborated"), h("dd", {}, fmt(heal.low_confidence_found)),
          h("dt", {}, "Mode"), h("dd", {}, heal.dry_run ? "report only — nothing changed" : "applied"))
      : h("p", { class: "muted small" }, "No healer report yet. It runs nightly, or from the Jobs panel."));
  } catch (error) {
    tiles.replaceChildren(h("p", {}, `Could not load the overview: ${error.message}`));
  }
}

// ---------------------------------------------------------------- wiring

function init() {
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  const question = $("question");
  question.addEventListener("input", () => { question.style.height = "auto"; question.style.height = `${question.scrollHeight}px`; });
  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); $("composer").requestSubmit(); }
  });
  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = question.value;
    question.value = "";
    question.style.height = "auto";
    ask(text);
  });
  $("new-session").addEventListener("click", newConversation);

  let searchTimer;
  $("doc-search").addEventListener("input", (event) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { state.docs.search = event.target.value.trim(); state.docs.offset = 0; loadDocs(); }, 300);
  });
  $("doc-status").addEventListener("change", (event) => { state.docs.status = event.target.value; state.docs.offset = 0; loadDocs(); });
  $("doc-prev").addEventListener("click", () => { state.docs.offset = Math.max(0, state.docs.offset - state.docs.limit); loadDocs(); });
  $("doc-next").addEventListener("click", () => { state.docs.offset += state.docs.limit; loadDocs(); });
  $("ingest-form").addEventListener("submit", submitIngest);
  $("run-communities").addEventListener("click", () => runJob("communities"));
  $("run-heal").addEventListener("click", () => runJob("heal-report"));

  refreshHealth();
  setInterval(refreshHealth, 60000);
  loadSuggestions();
  // Deep links: #corpus or #graph opens that tab; ?q=... asks a question on arrival.
  const tab = location.hash.slice(1);
  showTab(["ask", "corpus", "graph"].includes(tab) ? tab : "ask");
  window.addEventListener("hashchange", () => {
    const next = location.hash.slice(1);
    if (["ask", "corpus", "graph"].includes(next)) showTab(next);
  });
  const q = new URLSearchParams(location.search).get("q");
  if (q) ask(q);
}

init();
