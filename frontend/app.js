const $ = (s) => document.querySelector(s);
const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => (x * 100).toFixed(1) + " %";
let LABELS = {};

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    const e = new Error(msg); e.status = r.status; throw e;
  }
  return r.json();
}

function banner(msg) { const b = $("#banner"); b.hidden = !msg; b.textContent = msg || ""; }

async function waitReady() {
  for (;;) {
    try {
      const h = await api("/health");
      if (h.error) { banner("Error al iniciar el sistema: " + h.error); return false; }
      if (h.ready) { banner(""); return true; }
      banner("Cargando índices y modelos… la primera vez puede tardar hasta un minuto.");
    } catch (_) { banner("Conectando con el servidor…"); }
    await new Promise((r) => setTimeout(r, 2000));
  }
}

async function init() {
  if (!(await waitReady())) return;
  const models = await api("/api/models");
  LABELS = Object.fromEntries(models.map((m) => [m.id, m.label]));
  $("#model").innerHTML = models.map((m) => `<option value="${m.id}"${m.id === "bm25" ? " selected" : ""}>${esc(m.label)}</option>`).join("")
    + `<option value="all">Comparar todos los modelos</option>`;
  api("/api/sample-queries").then(renderSamples).catch(() => {});
  loadMetrics();
}

function renderSamples(samples) {
  $("#samples").innerHTML = samples.map((s, i) => `<button type="button" class="chip" data-i="${i}">${esc(s.query.length > 70 ? s.query.slice(0, 70) + "…" : s.query)}<small>${esc(s.note)}</small></button>`).join("");
  $("#samples").onclick = (e) => {
    const b = e.target.closest(".chip"); if (!b) return;
    $("#q").value = samples[b.dataset.i].query; setView("results"); run();
  };
}

function badges(r) {
  const b = [];
  if (r.relevant === true) b.push('<span class="pill ok">relevante (qrels)</span>');
  return b.join("");
}

function card(r, extra = "") {
  const cls = ["card", r.relevant === true ? "rel" : "", extra ? "shared" : ""].join(" ");
  return `<div class="${cls}" data-doc="${esc(r.doc_id)}">
    <div class="top"><span>#${r.rank} · doc ${esc(r.doc_id)}</span><span>score ${r.score}</span></div>
    ${r.title ? `<div class="title">${esc(r.title)}</div>` : ""}
    <div class="text">${esc(r.snippet)}</div>
    <div class="badges">${badges(r)}${extra}</div></div>`;
}

function queryMetrics(qm) {
  if (!qm) return "";
  return `<span class="pill">P@${qm.k} ${qm.precision.toFixed(2)}</span><span class="pill">R@${qm.k} ${qm.recall.toFixed(2)}</span>`
    + `<span class="pill">nDCG@${qm.k} ${qm.ndcg.toFixed(2)}</span><span class="pill">AP ${qm.ap.toFixed(2)}</span>`
    + `<span class="pill">${qm.n_relevant} relevantes en qrels</span>`;
}

function emptyMsg(run) { return run.results.length ? "" : '<p class="muted">Ningún documento comparte términos con la consulta.</p>'; }

async function run() {
  const q = $("#q").value.trim(); if (!q) return;
  const model = $("#model").value, k = $("#k").value, btn = $("#form button.primary");
  btn.disabled = true; setView("results");
  $("#view-results").innerHTML = '<p class="muted">Buscando…</p>';
  try {
    if (model === "all") renderCompare(await api(`/api/compare?q=${encodeURIComponent(q)}&k=${k}`));
    else renderSearch(await api(`/api/search?q=${encodeURIComponent(q)}&model=${model}&k=${k}`));
  } catch (e) {
    $("#view-results").innerHTML = `<p class="banner">${esc(e.message)}</p>`;
    if (e.status === 503) waitReady();
  } finally { btn.disabled = false; }
}

function renderSearch(d) {
  $("#view-results").innerHTML = `<div class="head"><h2>${esc(d.label)}</h2><span class="pill">${d.latency_ms} ms</span>${queryMetrics(d.query_metrics)}</div>`
    + emptyMsg(d)
    + (d.query_known ? "" : '<p class="note">Consulta libre: no hay qrels, así que no se calculan métricas ni se marcan documentos relevantes.</p>')
    + d.results.map((r) => card(r)).join("");
}

function renderCompare(d) {
  const ids = Object.keys(d.runs);
  const count = {};
  ids.forEach((m) => d.runs[m].results.forEach((r) => { (count[r.doc_id] ||= []).push(m); }));
  const cols = ids.map((m) => {
    const run = d.runs[m];
    return `<div class="col"><h3>${esc(run.label)}</h3>
      <div class="head"><span class="pill">${run.latency_ms} ms</span>${queryMetrics(run.query_metrics)}</div>${emptyMsg(run)}
      ${run.results.map((r) => {
        const others = count[r.doc_id].filter((x) => x !== m).map((x) => LABELS[x]);
        return card(r, others.length ? `<span class="pill">también en: ${esc(others.join(", "))}</span>` : "");
      }).join("")}</div>`;
  }).join("");
  const head = "<tr><th></th>" + ids.map((m) => `<th>${esc(LABELS[m])}</th>`).join("") + "</tr>";
  const rows = ids.map((a) => `<tr><td>${esc(LABELS[a])}</td>` + ids.map((b) => `<td>${a === b ? "—" : d.overlap[a][b] + " / " + d.k}</td>`).join("") + "</tr>").join("");
  $("#view-results").innerHTML = `<div class="legend">Borde naranja: documento recuperado por más de un modelo. Borde verde: relevante según los qrels.</div>
    <div class="cols">${cols}</div>
    <h3>Documentos en común entre los top-${d.k}</h3><div class="tablewrap"><table>${head}${rows}</table></div>
    <p class="note">Los scores de cada modelo tienen escalas distintas (BM25 no está acotado, el coseno va de 0 a 1): no se pueden comparar entre columnas. Compara posiciones y documentos recuperados.</p>`
    + (d.query_known ? "" : '<p class="note">Consulta libre: no hay qrels, así que no se calculan métricas ni se marcan documentos relevantes.</p>');
}

async function loadMetrics() {
  const box = $("#view-metrics");
  try {
    const d = await api("/api/metrics");
    const cols = [["precision@10", "Precision@10"], ["recall@10", "Recall@10"], ["ndcg@10", "nDCG@10"], ["recall@20", "Recall@20"], ["MAP@100", "MAP@100"]];
    const max = Object.fromEntries(cols.map(([k]) => [k, Math.max(...d.order.map((m) => d.models[m].summary[k]))]));
    const best = d.config.bm25 ? `BM25 con k1=${d.config.bm25.k1} y b=${d.config.bm25.b}` : "";
    const first = d.models[d.order[0]];
    box.innerHTML = `<div class="tablewrap"><table>
      <tr><th>Modelo</th>${cols.map(([, l]) => `<th>${l}</th>`).join("")}<th>Latencia (ms/consulta)</th></tr>
      ${d.order.map((m) => { const r = d.models[m]; return `<tr><td>${esc(r.label)}</td>${cols.map(([k]) => `<td>${r.summary[k].toFixed(4)}<span class="bar" style="width:${(r.summary[k] / max[k]) * 100}%"></span></td>`).join("")}<td>${r.latency_ms.toFixed(1)}</td></tr>`; }).join("")}
    </table></div>
    <p class="note">Evaluación sobre ${d.config.n_queries ?? first.n_queries} consultas del conjunto de <b>prueba</b> de FiQA-2018 con sus qrels. Precision y Recall se promedian por consulta; MAP@100 es el promedio de la precisión promedio sobre los 100 primeros resultados. Los parámetros (${esc(best)}) se eligieron en las consultas de desarrollo, no en las de prueba. La latencia de los embeddings incluye calcular el embedding de la consulta. Muchas consultas tienen pocos documentos relevantes (2,6 en promedio), por eso la Precision@10 máxima posible es baja.</p>`;
  } catch (e) { box.innerHTML = `<p class="muted">${esc(e.message)}</p>`; }
}

function setView(v) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
  $("#view-results").hidden = v !== "results"; $("#view-metrics").hidden = v !== "metrics";
}

async function openDoc(id) {
  try {
    const d = await api(`/api/doc/${encodeURIComponent(id)}`);
    $("#doc article").innerHTML = `<h3>${esc(d.title || "Documento " + d.doc_id)}</h3><p class="muted">doc ${esc(d.doc_id)}</p><p>${esc(d.text)}</p>`;
    $("#doc").showModal();
  } catch (e) { banner(e.message); }
}

$("#form").addEventListener("submit", (e) => { e.preventDefault(); run(); });
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
$("#doc-close").addEventListener("click", () => $("#doc").close());
$("#view-results").addEventListener("click", (e) => { const c = e.target.closest(".card"); if (c) openDoc(c.dataset.doc); });
$("#view-results").addEventListener("mouseover", (e) => {
  const c = e.target.closest(".card"); document.querySelectorAll(".card.hl").forEach((x) => x.classList.remove("hl"));
  if (c) document.querySelectorAll(`.card[data-doc="${CSS.escape(c.dataset.doc)}"]`).forEach((x) => x.classList.add("hl"));
});
init();
