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
  sliders: '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
  chevron: '<path d="m6 9 6 6 6-6"/>',
  chart: '<path d="M3 3v18h18"/><path d="m7 15 4-4 3 3 5-6"/>',
};
const icon = (n) => `<svg class="i" viewBox="0 0 24 24" aria-hidden="true">${ICON[n]}</svg>`;
const SHORT = { jaccard: "Jaccard", tfidf: "TF-IDF", bm25: "BM25", embeddings: "Embeddings" };
const MODEL_COLOR = { jaccard: "var(--chart-1)", tfidf: "var(--chart-2)", bm25: "var(--chart-3)", embeddings: "var(--chart-4)" };
const METRICS = [["MAP@100", "MAP@100"], ["ndcg@10", "nDCG@10"], ["precision@10", "Precision@10"], ["recall@10", "Recall@10"], ["recall@20", "Recall@20"]];
let LABELS = {}, compare = null, compareMode = "columns", metricsData = null, analysisData = null, chartMetric = "MAP@100", state = "home", ready = false;

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
const isDark = () => (document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches);
function applyThemeIcon() { $("#theme").innerHTML = icon(isDark() ? "sun" : "moon"); }
$("#theme").addEventListener("click", () => {
  const next = isDark() ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (_) {}
  applyThemeIcon();
});

/* ---------- navegación: portada → resultados → análisis ---------- */
function closeAdv() { $("#adv").hidden = true; $("#adv-toggle").setAttribute("aria-expanded", "false"); }
function setState(s, url = null) {
  state = s; document.body.dataset.state = s;
  $("#hero").hidden = s !== "home"; $("#view-results").hidden = s !== "results"; $("#view-metrics").hidden = s !== "analysis";
  const slot = s === "home" ? $("#hero-slot") : $("#bar-slot");
  if ($("#searchwrap").parentElement !== slot) slot.appendChild($("#searchwrap"));
  $("#nav-analysis").classList.toggle("active", s === "analysis");
  closeAdv();
  if (url !== null) history.pushState({ s }, "", url);
  window.scrollTo(0, 0);
}
const homeUrl = () => location.pathname;
const resultsUrl = (q, model, k) => `${location.pathname}?${new URLSearchParams({ q, model, k })}`;
const analysisUrl = () => `${location.pathname}#analisis`;

async function route() {
  const p = new URLSearchParams(location.search), q = p.get("q");
  if (q && ready) {
    $("#q").value = q;
    if ([...$("#model").options].some((o) => o.value === p.get("model"))) $("#model").value = p.get("model");
    if ([...$("#k").options].some((o) => o.value === p.get("k"))) $("#k").value = p.get("k");
    await run(false);
  } else if (location.hash === "#analisis" && ready) await showAnalysis(false);
  else setState("home");
}
window.addEventListener("popstate", route);
$("#brand").addEventListener("click", (e) => { e.preventDefault(); setState("home", homeUrl()); $("#q").focus(); });
$("#nav-analysis").addEventListener("click", () => showAnalysis(true));

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
  $("#adv-toggle").innerHTML = icon("sliders") + '<span class="toggle-label">Ajustes avanzados</span>' + icon("chevron");
  $("#nav-analysis").innerHTML = icon("chart") + "Análisis";
  $("#doc-close").innerHTML = icon("x");
  setState("home"); $("#q").focus();
  if (!(await waitReady())) return;
  const models = await api("/api/models");
  LABELS = Object.fromEntries(models.map((m) => [m.id, m.label]));
  $("#model").innerHTML = `<option value="all" selected>Comparar todos los modelos</option>`
    + models.map((m) => `<option value="${m.id}">${esc(m.label)}</option>`).join("");
  ready = true;
  api("/api/sample-queries").then(renderSamples).catch(() => {});
  route();
}
function renderSamples(samples) {
  $("#samples").innerHTML = samples.map((s, i) => `<button type="button" class="chip" data-i="${i}">${esc(s.query.length > 64 ? s.query.slice(0, 64) + "…" : s.query)}<small>${esc(s.note)}</small></button>`).join("");
  $("#samples").onclick = (e) => {
    const b = e.target.closest(".chip"); if (!b) return;
    $("#q").value = samples[b.dataset.i].query; closeAdv(); run();
  };
}
$("#adv-toggle").addEventListener("click", () => {
  const open = $("#adv").hidden; $("#adv").hidden = !open; $("#adv-toggle").setAttribute("aria-expanded", String(open));
});
document.addEventListener("click", (e) => { if (state !== "home" && !$("#adv").hidden && !e.target.closest("#searchwrap")) closeAdv(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeAdv(); });

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
const resTitle = (q, k) => `<div class="res-title">Resultados para <b>«${esc(q)}»</b> · top-${k}</div>`;
function skeletons() { $("#view-results").innerHTML = '<div class="results"><div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div></div>'; }

/* ---------- búsqueda ---------- */
async function run(push = true) {
  const q = $("#q").value.trim(); if (!q) return;
  const model = $("#model").value, k = $("#k").value, btn = $("#go");
  setState("results", push ? resultsUrl(q, model, k) : null);
  btn.disabled = true; skeletons();
  try {
    if (model === "all") { compare = await api(`/api/compare?q=${encodeURIComponent(q)}&k=${k}`); renderCompare(); }
    else renderSearch(await api(`/api/search?q=${encodeURIComponent(q)}&model=${model}&k=${k}`), q);
  } catch (e) {
    $("#view-results").innerHTML = `<div class="alert destructive">${esc(e.message)}</div>`;
    if (e.status === 503) waitReady();
  } finally { btn.disabled = false; }
}
function renderSearch(d, q) {
  $("#view-results").innerHTML = resTitle(q, d.k) + `<div class="head"><h2>${esc(d.label)}</h2><span class="badge outline">${d.latency_ms} ms</span>${queryBadges(d.query_metrics)}</div>`
    + (d.query_known ? "" : freeNote)
    + (d.results.length ? `<div class="results">${d.results.map((r) => result(r)).join("")}</div>` : noHits);
}
function renderCompare() {
  const d = compare, ids = Object.keys(d.runs), who = {};
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
  $("#view-results").innerHTML = resTitle(d.query, d.k) + toolbar + body
    + `<div class="card" style="margin-top:16px"><div class="card-title">Documentos en común entre los top-${d.k}</div><p class="card-desc">Cuántos resultados comparten dos modelos. Los scores no se comparan porque cada modelo usa una escala distinta (BM25 no está acotado y el coseno va de 0 a 1).</p><div class="tablewrap"><table>${head}${overlap}</table></div></div>`
    + (d.query_known ? "" : freeNote);
}

/* ---------- análisis ---------- */
async function showAnalysis(push = true) {
  setState("analysis", push ? analysisUrl() : null);
  $("#view-metrics").innerHTML = '<div class="results"><div class="skeleton"></div><div class="skeleton"></div></div>';
  const [m, a] = await Promise.allSettled([metricsData ? metricsData : api("/api/metrics"), analysisData ? analysisData : api("/api/analysis")]);
  if (m.status === "fulfilled") metricsData = m.value;
  if (a.status === "fulfilled") analysisData = a.value;
  renderAnalysis(m.status === "rejected" ? m.reason.message : null, a.status === "rejected" ? a.reason.message : null);
}
function metricsSection(d) {
  const order = d.order, cfg = d.config || {};
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
  return `<div class="card"><div class="card-title">Resultados en el conjunto de prueba</div>
      <p class="card-desc">Precision y Recall se promedian por consulta; MAP@100 es el promedio de la precisión promedio sobre los 100 primeros resultados. En negrita, el mejor valor de cada columna.</p>${table}</div>
    <div class="grid2"><div class="card"><div class="toolbar"><div><div class="card-title">Comparación por métrica</div><p class="card-desc" style="margin:0">Mayor es mejor.</p></div>
        <div class="seg" role="group" aria-label="Métrica">${METRICS.map(([k, l]) => `<button data-metric="${k}" class="${k === chartMetric ? "on" : ""}">${l}</button>`).join("")}</div></div>${bars}</div>
      <div class="card"><div class="card-title">Configuración</div><p class="card-desc">Parámetros elegidos en las consultas de desarrollo, no en las de prueba.</p><dl class="kv">${kv.map(([x, y]) => `<dt>${x}</dt><dd>${esc(y)}</dd>`).join("")}</dl></div></div>`;
}
function comparisonSection(a) {
  const A = a.labels.a, B = a.labels.b, f = (x) => x.toFixed(4), sgn = (x) => (x >= 0 ? "+" : "") + x.toFixed(4);
  const tot = a.a_gana + a.b_gana + a.empates;
  const maxv = Math.max(...Object.values(a.buckets).flatMap((b) => [b.a, b.b]), a.mean_a, a.mean_b);
  const rows = Object.entries(a.buckets).map(([name, b]) => `<tr><td>${esc(name)}</td><td>${b.n}</td>
    <td><span class="mini"><span style="width:${(b.a / maxv) * 100}%;background:var(--chart-4)"></span></span>${f(b.a)}</td>
    <td><span class="mini"><span style="width:${(b.b / maxv) * 100}%;background:var(--chart-3)"></span></span>${f(b.b)}</td>
    <td class="${b.a - b.b >= 0 ? "pos" : "neg"}">${sgn(b.a - b.b)}</td></tr>`).join("");
  const list = (items, title) => `<div><div class="subtitle">${title}</div><div class="qlist">${items.map((r) => `<button type="button" data-q="${esc(r.query)}"><span>${esc(r.query.length > 90 ? r.query.slice(0, 90) + "…" : r.query)}</span><small>${esc(SHORT.embeddings)} ${r.a.toFixed(2)} · ${esc(SHORT.bm25)} ${r.b.toFixed(2)}</small></button>`).join("")}</div></div>`;
  return `<div class="card"><div class="card-title">${esc(A)} frente a ${esc(B)}</div>
    <p class="card-desc">Comparación consulta por consulta con la precisión promedio (AP) en las ${a.n} consultas de prueba. Pulsa una consulta para verla en la comparación.</p>
    <div class="tiles">
      <div class="tile"><div class="k">AP medio</div><div class="v">${f(a.mean_a)} <span class="muted small">vs</span> ${f(a.mean_b)}</div><div class="s">${esc(SHORT.embeddings)} · ${esc(SHORT.bm25)}</div></div>
      <div class="tile"><div class="k">Diferencia media</div><div class="v ${a.diff_media >= 0 ? "pos" : "neg"}">${sgn(a.diff_media)}</div><div class="s">IC 95 %: ${sgn(a.ic95[0])} a ${sgn(a.ic95[1])}</div></div>
      <div class="tile"><div class="k">Consultas ganadas</div><div class="v">${a.a_gana} <span class="muted small">/</span> ${a.b_gana}</div>
        <div class="stack"><span style="width:${(a.a_gana / tot) * 100}%;background:var(--chart-4)"></span><span style="width:${(a.b_gana / tot) * 100}%;background:var(--chart-3)"></span></div><div class="s">${esc(SHORT.embeddings)} · ${esc(SHORT.bm25)} · ${a.empates} empates</div></div>
    </div>
    <div class="subtitle" style="margin-top:0">Por tipo de consulta</div>
    <div class="tablewrap"><table><tr><th>Tipo</th><th>n</th><th>${esc(SHORT.embeddings)}</th><th>${esc(SHORT.bm25)}</th><th>Diferencia</th></tr>${rows}</table></div>
    <div class="cols2">${list(a.a_mejora_mas, esc(SHORT.embeddings) + " mejora más")}${list(a.a_empeora_mas, esc(SHORT.embeddings) + " empeora más")}</div></div>`;
}
function renderAnalysis(metricsErr, analysisErr) {
  $("#view-metrics").innerHTML = `<div class="page-head"><h2>Análisis</h2><p class="muted">Métricas de evaluación de los cuatro modelos y comparación entre la recuperación semántica y BM25.</p></div>`
    + (metricsData ? metricsSection(metricsData) : `<div class="alert info">${esc(metricsErr || "Sin métricas.")}</div>`)
    + `<div style="margin-top:16px">${analysisData ? comparisonSection(analysisData) : `<div class="alert info">${esc(analysisErr || "Sin análisis.")}</div>`}</div>`
    + `<p class="muted small" style="margin-top:12px">Las consultas tienen pocos documentos relevantes (2,6 en promedio), por eso la Precision@10 máxima posible es baja. La latencia de los embeddings incluye calcular el embedding de la consulta. Los qrels de FiQA son incompletos: un documento sin juicio no es necesariamente irrelevante.</p>`;
}

/* ---------- eventos ---------- */
async function openDoc(id) {
  try {
    const d = await api(`/api/doc/${encodeURIComponent(id)}`);
    $("#doc-title").textContent = d.title || "Documento " + d.doc_id; $("#doc-id").textContent = "doc " + d.doc_id; $("#doc-text").textContent = d.text;
    $("#doc").showModal();
  } catch (e) { alertBox("destructive", e.message); }
}
$("#form").addEventListener("submit", (e) => { e.preventDefault(); run(); });
$("#doc-close").addEventListener("click", () => $("#doc").close());
$("#doc").addEventListener("click", (e) => { if (e.target === $("#doc")) $("#doc").close(); });
$("#view-results").addEventListener("click", (e) => {
  const m = e.target.closest("[data-mode]");
  if (m) { compareMode = m.dataset.mode; renderCompare(); return; }
  const c = e.target.closest("[data-doc]"); if (c) openDoc(c.dataset.doc);
});
$("#view-metrics").addEventListener("click", (e) => {
  const b = e.target.closest("[data-metric]");
  if (b) { chartMetric = b.dataset.metric; renderAnalysis(); return; }
  const q = e.target.closest("[data-q]");
  if (q) { $("#q").value = q.dataset.q; $("#model").value = "all"; run(); }
});
$("#view-results").addEventListener("mouseover", (e) => {
  document.querySelectorAll(".hl").forEach((x) => x.classList.remove("hl"));
  const c = e.target.closest(".result[data-doc]");
  if (c) document.querySelectorAll(`.result[data-doc="${CSS.escape(c.dataset.doc)}"]`).forEach((x) => x.classList.add("hl"));
});
init();
