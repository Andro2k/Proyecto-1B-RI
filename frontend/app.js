const $ = (s) => document.querySelector(s);
const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const ICON = {
  search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4"/>',
  moon: '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
  columns: '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M12 3v18"/>',
  table: '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18M3 15h18"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
};
const icon = (n) => `<svg class="i" viewBox="0 0 24 24" aria-hidden="true">${ICON[n]}</svg>`;
const SHORT = { jaccard: "Jaccard", tfidf: "TF-IDF", bm25: "BM25", embeddings: "Embeddings" };
const MODEL_COLOR = { jaccard: "var(--chart-1)", tfidf: "var(--chart-2)", bm25: "var(--chart-3)", embeddings: "var(--chart-4)" };
const METRICS = [["MAP@100", "MAP@100"], ["ndcg@10", "nDCG@10"], ["precision@10", "Precision@10"], ["recall@10", "Recall@10"], ["recall@20", "Recall@20"]];
let LABELS = {}, compare = null, compareMode = "columns", metricsData = null, chartMetric = "MAP@100";

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    const e = new Error(msg); e.status = r.status; throw e;
  }
  return r.json();
}
function alertBox(kind, msg) { $("#alerts").innerHTML = msg ? `<div class="alert ${kind}">${esc(msg)}</div>` : ""; }

/* ---------- tema ---------- */
function applyThemeIcon() {
  const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  $("#theme").innerHTML = icon(dark ? "sun" : "moon");
}
$("#theme").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  const next = dark ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (_) {}
  applyThemeIcon();
});

/* ---------- arranque ---------- */
async function waitReady() {
  for (;;) {
    try {
      const h = await api("/health");
      if (h.error) { alertBox("destructive", "Error al iniciar el sistema: " + h.error); return false; }
      if (h.ready) { alertBox("warning", (h.warnings || []).join(" ")); return true; }
      alertBox("info", "Cargando índices y modelos… la primera vez puede tardar hasta un minuto.");
    } catch (_) { alertBox("info", "Conectando con el servidor…"); }
    await new Promise((r) => setTimeout(r, 2000));
  }
}
async function init() {
  applyThemeIcon();
  $("#go").innerHTML = icon("search") + "Buscar";
  $("#doc-close").innerHTML = icon("x");
  showEmpty();
  if (!(await waitReady())) return;
  const models = await api("/api/models");
  LABELS = Object.fromEntries(models.map((m) => [m.id, m.label]));
  $("#model").innerHTML = models.map((m) => `<option value="${m.id}"${m.id === "bm25" ? " selected" : ""}>${esc(m.label)}</option>`).join("")
    + `<option value="all">Comparar todos los modelos</option>`;
  api("/api/sample-queries").then(renderSamples).catch(() => {});
  loadMetrics();
}
function renderSamples(samples) {
  $("#samples").innerHTML = samples.map((s, i) => `<button type="button" class="chip" data-i="${i}">${esc(s.query.length > 64 ? s.query.slice(0, 64) + "…" : s.query)}<small>${esc(s.note)}</small></button>`).join("");
  $("#samples").onclick = (e) => {
    const b = e.target.closest(".chip"); if (!b) return;
    $("#q").value = samples[b.dataset.i].query; setView("results"); run();
  };
}
function showEmpty() {
  $("#view-results").innerHTML = '<div class="empty"><strong>Aún no hay resultados</strong>Escribe una consulta o elige uno de los ejemplos para ver el ranking de cada modelo.</div>';
}

/* ---------- piezas ---------- */
function queryBadges(qm) {
  if (!qm) return "";
  return `<span class="badge">P@${qm.k} ${qm.precision.toFixed(2)}</span><span class="badge">R@${qm.k} ${qm.recall.toFixed(2)}</span>`
    + `<span class="badge">nDCG@${qm.k} ${qm.ndcg.toFixed(2)}</span><span class="badge">AP ${qm.ap.toFixed(2)}</span>`
    + `<span class="badge outline">${qm.n_relevant} relevantes en qrels</span>`;
}
function result(r, others = []) {
  const cls = ["result", r.relevant === true ? "rel" : "", others.length ? "shared" : ""].join(" ");
  const badges = (r.relevant === true ? `<span class="badge success">${icon("check")}Relevante (qrels)</span>` : "")
    + (others.length ? `<span class="badge shared">También en: ${esc(others.join(", "))}</span>` : "");
  return `<button type="button" class="${cls}" data-doc="${esc(r.doc_id)}">
    <span class="rank">${r.rank}</span>
    <span><span class="meta" data-rank="#${r.rank}"><span>doc ${esc(r.doc_id)}</span><span>score ${r.score}</span></span>
    ${r.title ? `<span class="title" style="display:block">${esc(r.title)}</span>` : ""}
    <span class="text">${esc(r.snippet)}</span>
    ${badges ? `<span class="badges">${badges}</span>` : ""}</span></button>`;
}
const freeNote = '<div class="alert info">Consulta libre: no hay qrels para ella, así que no se calculan métricas ni se marcan documentos relevantes.</div>';
const noHits = '<div class="empty"><strong>Sin resultados</strong>Ningún documento comparte términos con la consulta.</div>';
function skeletons() { $("#view-results").innerHTML = '<div class="results"><div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div></div>'; }

/* ---------- búsqueda ---------- */
async function run() {
  const q = $("#q").value.trim(); if (!q) return;
  const model = $("#model").value, k = $("#k").value, btn = $("#go");
  btn.disabled = true; setView("results"); skeletons();
  try {
    if (model === "all") { compare = await api(`/api/compare?q=${encodeURIComponent(q)}&k=${k}`); renderCompare(); }
    else renderSearch(await api(`/api/search?q=${encodeURIComponent(q)}&model=${model}&k=${k}`));
  } catch (e) {
    $("#view-results").innerHTML = `<div class="alert destructive">${esc(e.message)}</div>`;
    if (e.status === 503) waitReady();
  } finally { btn.disabled = false; }
}
function renderSearch(d) {
  $("#view-results").innerHTML = `<div class="head"><h2>${esc(d.label)}</h2><span class="badge outline">${d.latency_ms} ms</span>${queryBadges(d.query_metrics)}</div>`
    + (d.query_known ? "" : freeNote)
    + (d.results.length ? `<div class="results">${d.results.map((r) => result(r)).join("")}</div>` : noHits);
}
function renderCompare() {
  const d = compare, ids = Object.keys(d.runs);
  const who = {};
  ids.forEach((m) => d.runs[m].results.forEach((r) => { (who[r.doc_id] ||= []).push(m); }));
  const others = (m, doc) => who[doc].filter((x) => x !== m).map((x) => SHORT[x] || x);
  const toolbar = `<div class="toolbar"><div class="legend"><span><i class="dot" style="background:var(--success)"></i>Relevante según qrels</span><span><i class="dot" style="background:var(--shared)"></i>Recuperado por más de un modelo</span></div>
    <div class="seg" role="group" aria-label="Tipo de vista"><button data-mode="columns" class="${compareMode === "columns" ? "on" : ""}">${icon("columns")}Columnas</button><button data-mode="table" class="${compareMode === "table" ? "on" : ""}">${icon("table")}Posiciones</button></div></div>`;
  let body;
  if (compareMode === "columns") {
    body = `<div class="cols">${ids.map((m) => {
      const run = d.runs[m];
      return `<div class="col"><h3>${esc(run.label)}</h3><div class="head"><span class="badge outline">${run.latency_ms} ms</span>${queryBadges(run.query_metrics)}</div>
        <div class="results">${run.results.length ? run.results.map((r) => result(r, others(m, r.doc_id))).join("") : noHits}</div></div>`;
    }).join("")}</div>`;
  } else {
    const rows = Array.from({ length: d.k }, (_, i) => `<tr><td>${i + 1}</td>${ids.map((m) => {
      const r = d.runs[m].results[i];
      if (!r) return "<td></td>";
      const cls = [r.relevant === true ? "rel" : "", others(m, r.doc_id).length ? "shared" : ""].join(" ");
      return `<td class="${cls}" data-doc="${esc(r.doc_id)}"><b>doc ${esc(r.doc_id)}</b><br><span class="muted">${esc((r.title || r.snippet).slice(0, 90))}</span></td>`;
    }).join("")}</tr>`).join("");
    body = `<div class="tablewrap"><table class="cmp"><tr><th>#</th>${ids.map((m) => `<th>${esc(LABELS[m])}<br><span class="badge outline">${d.runs[m].latency_ms} ms</span></th>`).join("")}</tr>${rows}</table></div>
      <div class="head" style="margin-top:12px">${ids.map((m) => queryBadges(d.runs[m].query_metrics) ? `<span class="muted small">${esc(LABELS[m])}:</span>${queryBadges(d.runs[m].query_metrics)}` : "").join("")}</div>`;
  }
  const head = "<tr><th></th>" + ids.map((m) => `<th>${esc(LABELS[m])}</th>`).join("") + "</tr>";
  const overlap = ids.map((a) => `<tr><td>${esc(LABELS[a])}</td>` + ids.map((b) => `<td>${a === b ? "—" : d.overlap[a][b] + " / " + d.k}</td>`).join("") + "</tr>").join("");
  $("#view-results").innerHTML = toolbar + body
    + `<div class="card" style="margin-top:16px"><div class="card-title">Documentos en común entre los top-${d.k}</div><p class="card-desc">Cuántos resultados comparten dos modelos. Los scores no se comparan porque cada modelo usa una escala distinta (BM25 no está acotado y el coseno va de 0 a 1).</p><div class="tablewrap"><table>${head}${overlap}</table></div></div>`
    + (d.query_known ? "" : freeNote);
}

/* ---------- métricas ---------- */
async function loadMetrics() {
  try { metricsData = await api("/api/metrics"); renderMetrics(); }
  catch (e) { $("#view-metrics").innerHTML = `<div class="alert info">${esc(e.message)}</div>`; }
}
function renderMetrics() {
  const d = metricsData, order = d.order, cfg = d.config || {};
  const best = Object.fromEntries(METRICS.map(([k]) => [k, Math.max(...order.map((m) => d.models[m].summary[k]))]));
  const table = `<div class="tablewrap"><table><tr><th>Modelo</th>${METRICS.map(([, l]) => `<th>${l}</th>`).join("")}<th>ms / consulta</th></tr>
    ${order.map((m) => { const r = d.models[m]; return `<tr><td>${esc(r.label)}</td>${METRICS.map(([k]) => `<td class="${r.summary[k] === best[k] ? "best" : ""}">${r.summary[k].toFixed(4)}</td>`).join("")}<td>${r.latency_ms.toFixed(1)}</td></tr>`; }).join("")}</table></div>`;
  const max = Math.max(...order.map((m) => d.models[m].summary[chartMetric]));
  const bars = `<div class="bars">${order.map((m) => { const v = d.models[m].summary[chartMetric];
    return `<div class="bar-row"><span>${esc(d.models[m].label)}</span><div class="bar-track"><div class="bar-fill" style="width:${(v / max) * 100}%;background:${MODEL_COLOR[m]}"></div></div><span class="bar-val">${v.toFixed(4)}</span></div>`; }).join("")}</div>`;
  const kv = [["Consultas de prueba", cfg.n_queries ?? d.models[order[0]].n_queries],
    ["BM25", cfg.bm25 ? `k1 = ${cfg.bm25.k1}, b = ${cfg.bm25.b}` : "—"],
    ["TF-IDF", cfg.tfidf ? `TF ${cfg.tfidf.tf}, IDF ${cfg.tfidf.idf}` : "—"],
    ["Índice invertido", cfg.index_mb ? `${cfg.index_mb} MB` : "—"]];
  $("#view-metrics").innerHTML = `<div class="card"><div class="card-title">Resultados en el conjunto de prueba</div>
      <p class="card-desc">Precision y Recall se promedian por consulta; MAP@100 es el promedio de la precisión promedio sobre los 100 primeros resultados. En negrita, el mejor valor de cada columna.</p>${table}</div>
    <div class="grid2"><div class="card"><div class="toolbar"><div><div class="card-title">Comparación por métrica</div><p class="card-desc" style="margin:0">Mayor es mejor.</p></div>
        <div class="seg" role="group" aria-label="Métrica">${METRICS.map(([k, l]) => `<button data-metric="${k}" class="${k === chartMetric ? "on" : ""}">${l}</button>`).join("")}</div></div>${bars}</div>
      <div class="card"><div class="card-title">Configuración</div><p class="card-desc">Parámetros elegidos en las consultas de desarrollo, no en las de prueba.</p><dl class="kv">${kv.map(([a, b]) => `<dt>${a}</dt><dd>${esc(b)}</dd>`).join("")}</dl></div></div>
    <p class="muted small" style="margin-top:12px">Las consultas tienen pocos documentos relevantes (2,6 en promedio), por eso la Precision@10 máxima posible es baja. La latencia de los embeddings incluye calcular el embedding de la consulta. Los qrels de FiQA son incompletos: un documento sin juicio no es necesariamente irrelevante.</p>`;
}

/* ---------- vistas y eventos ---------- */
function setView(v) {
  document.querySelectorAll(".tab").forEach((b) => { const on = b.dataset.view === v; b.classList.toggle("active", on); b.setAttribute("aria-selected", on); });
  $("#view-results").hidden = v !== "results"; $("#view-metrics").hidden = v !== "metrics";
}
async function openDoc(id) {
  try {
    const d = await api(`/api/doc/${encodeURIComponent(id)}`);
    $("#doc-title").textContent = d.title || "Documento " + d.doc_id; $("#doc-id").textContent = "doc " + d.doc_id; $("#doc-text").textContent = d.text;
    $("#doc").showModal();
  } catch (e) { alertBox("destructive", e.message); }
}
$("#form").addEventListener("submit", (e) => { e.preventDefault(); run(); });
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
$("#doc-close").addEventListener("click", () => $("#doc").close());
$("#doc").addEventListener("click", (e) => { if (e.target === $("#doc")) $("#doc").close(); });
$("#view-results").addEventListener("click", (e) => {
  const m = e.target.closest("[data-mode]");
  if (m) { compareMode = m.dataset.mode; renderCompare(); return; }
  const c = e.target.closest("[data-doc]"); if (c) openDoc(c.dataset.doc);
});
$("#view-metrics").addEventListener("click", (e) => { const b = e.target.closest("[data-metric]"); if (b) { chartMetric = b.dataset.metric; renderMetrics(); } });
$("#view-results").addEventListener("mouseover", (e) => {
  document.querySelectorAll(".hl").forEach((x) => x.classList.remove("hl"));
  const c = e.target.closest(".result[data-doc]");
  if (c) document.querySelectorAll(`.result[data-doc="${CSS.escape(c.dataset.doc)}"]`).forEach((x) => x.classList.add("hl"));
});
init();
