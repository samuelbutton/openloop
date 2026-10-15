import { $, short, fmt, el, svg, api, table } from "/dom.js";
import { curveView } from "/curves.js";

const state = { data: null, selected: null, hidden: new Set(), comparison: 0 };
const curves = curveView(state, selectRun);
function selectRun(id) {
  const run = state.data.nodes.find((node) => node.run_id === id);
  if (!run) return;
  state.selected = id;
  document
    .querySelectorAll(".node")
    .forEach((node) =>
      node.classList.toggle("selected", node.dataset.run === id),
    );
  const panel = $("inspector");
  panel.replaceChildren(
    el("p", "RUN INSPECTOR", "eyebrow"),
    el("h3", short(id)),
  );
  const dl = el("dl");
  const fields = {
    Status: run.status,
    Candidate: short(run.candidate_hash),
    Phase: run.phase,
    Fidelity: run.fidelity,
    Seed: run.seed,
    Work: `${fmt(run.budget)} ${run.budget_unit}`,
    Evidence: run.reused_from
      ? `Cache of ${short(run.reused_from)}`
      : "Original attempt",
    Mode:
      run.execution.simulated === true ? "Simulated" : "See execution settings",
  };
  for (const [name, value] of Object.entries(fields))
    dl.append(el("dt", name), el("dd", value));
  panel.append(dl);
  for (const [name, metric] of Object.entries(run.metrics))
    panel.append(el("p", `${name}: ${fmt(metric.value)} ${metric.unit}`));
  const times = Object.entries(run.stage_seconds)
    .map(([name, value]) => `${name} ${fmt(value)}s`)
    .join(" · ");
  if (times) panel.append(el("p", times));
  if (run.hypothesis) panel.append(el("p", run.hypothesis));
  for (const target of ["baseline", "candidate"]) {
    const button = el("button", `Use as ${target}`);
    button.addEventListener("click", () => {
      $(target).value = id;
      $("diff").scrollIntoView({ block: "start" });
      compare();
    });
    panel.append(button);
  }
  const details = el("details"),
    pre = el("pre", JSON.stringify(run.inputs, null, 2));
  details.append(el("summary", "Recorded inputs"), pre);
  panel.append(details);
}
function drawGraph() {
  const { nodes, edges } = state.data,
    holder = $("graph");
  $("graph-count").textContent = `${nodes.length} nodes`;
  holder.replaceChildren();
  if (!nodes.length) {
    holder.append(
      el(
        "p",
        "No selection runs yet. Refresh after an experiment is recorded.",
        "empty",
      ),
    );
    return;
  }
  const positions = new Map(),
    columns = new Map();
  for (const node of nodes) {
    const parents = edges
      .filter((edge) => edge.target === node.run_id)
      .map((edge) => positions.get(edge.source))
      .filter(Boolean);
    const depth = parents.length
      ? Math.max(...parents.map((point) => point.depth)) + 1
      : 0;
    const row = columns.get(depth) || 0;
    columns.set(depth, row + 1);
    positions.set(node.run_id, {
      depth,
      x: 18 + depth * 190,
      y: 18 + row * 83,
    });
  }
  const width = Math.max(430, columns.size * 190 + 20),
    height = Math.max(235, Math.max(...columns.values()) * 83 + 20);
  const drawing = svg("svg", {
    width,
    height,
    role: "group",
    "aria-label": "Experiment DAG",
  });
  for (const edge of edges) {
    const a = positions.get(edge.source),
      b = positions.get(edge.target);
    drawing.append(
      svg("path", {
        d: `M${a.x + 150},${a.y + 28} C${a.x + 173},${a.y + 28} ${b.x - 23},${b.y + 28} ${b.x},${b.y + 28}`,
        class: `edge ${edge.kind}`,
      }),
    );
  }
  for (const run of nodes) {
    const { x, y } = positions.get(run.run_id);
    const node = svg("g", {
      class: "node",
      transform: `translate(${x},${y})`,
      tabindex: 0,
      role: "button",
      "aria-label": `Run ${run.run_id}, ${run.status}`,
    });
    node.dataset.run = run.run_id;
    node.append(
      svg("rect", { width: 150, height: 58, rx: 7 }),
      svg("circle", { cx: 15, cy: 19, r: 3.5, class: run.status }),
      svg("text", { x: 26, y: 23, class: "run-id" }, short(run.run_id)),
      svg(
        "text",
        { x: 15, y: 43, class: "run-label" },
        `${run.fidelity} · seed ${run.seed}${run.reused_from ? " · cache" : ""}`,
      ),
      svg(
        "title",
        {},
        `${run.run_id}\n${run.status}\nCandidate ${run.candidate_hash}`,
      ),
    );
    node.addEventListener("click", () => selectRun(run.run_id));
    node.addEventListener("keydown", (event) => {
      if (["Enter", " "].includes(event.key)) {
        event.preventDefault();
        selectRun(run.run_id);
      }
    });
    drawing.append(node);
  }
  holder.append(drawing);
  if (state.selected) selectRun(state.selected);
}
function runOptions() {
  for (const id of ["baseline", "candidate"]) {
    const select = $(id),
      old = select.value;
    select.replaceChildren();
    for (const run of state.data.nodes) {
      const option = el(
        "option",
        `${short(run.run_id)} · ${run.fidelity} · seed ${run.seed}${run.reused_from ? " · cache" : ""}`,
      );
      option.value = run.run_id;
      select.append(option);
    }
    if (state.data.nodes.some((run) => run.run_id === old)) select.value = old;
    else if (id === "candidate" && select.options.length > 1)
      select.selectedIndex = select.options.length - 1;
  }
  $("compare").disabled = !state.data.nodes.length;
}
async function compare() {
  const baseline = $("baseline").value,
    candidate = $("candidate").value,
    token = ++state.comparison;
  if (!baseline || !candidate) return;
  const holder = $("diff-result");
  holder.replaceChildren(el("p", "Loading run comparison…", "empty"));
  try {
    const report = await api(
      `/api/diff?${new URLSearchParams({ baseline, candidate })}`,
    );
    if (token !== state.comparison) return;
    holder.replaceChildren();
    const cards = el("div", null, "diff-metrics");
    for (const metric of report.metrics) {
      const card = el("div", null, "metric-card");
      card.append(
        el("h3", metric.name),
        el(
          "strong",
          metric.comparable
            ? `${metric.delta > 0 ? "+" : ""}${fmt(metric.delta)}`
            : "Not comparable",
        ),
        el(
          "p",
          `${fmt(metric.baseline?.value)} ${metric.baseline?.unit || ""} → ${fmt(metric.candidate?.value)} ${metric.candidate?.unit || ""}`,
        ),
        el(
          "p",
          metric.comparable
            ? `Improvement-oriented gain ${fmt(metric.gain)}`
            : "Requires matching context, seed, metric metadata, and two distinct attempts.",
        ),
      );
      cards.append(card);
    }
    if (!report.metrics.length)
      cards.append(
        el("p", "No completed validation metrics for these runs.", "empty"),
      );
    holder.append(
      cards,
      el(
        "p",
        `${report.baseline.status} → ${report.candidate.status} · ${report.independent_attempts ? "Distinct execution attempts" : "Same execution attempt"}. No verification claim.`,
        "diff-note",
      ),
    );
    const changes = Object.entries(report.input_changes).map(
      ([name, value]) => [
        name,
        el("pre", JSON.stringify(value.baseline, null, 2)),
        el("pre", JSON.stringify(value.candidate, null, 2)),
      ],
    );
    if (changes.length) {
      const wrap = el("div", null, "table-wrap");
      wrap.append(table(["Input", "Baseline", "Candidate"], changes));
      holder.append(wrap);
    } else
      holder.append(el("p", "Recorded inputs are identical.", "diff-note"));
  } catch (error) {
    if (token === state.comparison)
      holder.replaceChildren(el("p", error.message, "empty"));
  }
}
function clearComparison() {
  ++state.comparison;
  $("diff-result").replaceChildren(
    el(
      "p",
      "Choose two runs to compare their inputs and validation scores.",
      "empty",
    ),
  );
}
async function refresh() {
  $("refresh").disabled = true;
  try {
    state.data = await api("/api/snapshot");
    $("ledger-name").textContent = state.data.ledger;
    $("updated").textContent = `Read ${new Date().toLocaleTimeString()}`;
    for (const [key, value] of Object.entries(state.data.summary))
      $(`stat-${key}`).textContent = fmt(value);
    const warnings = [];
    if (state.data.graph_truncated)
      warnings.push(
        `Graph shows the latest ${state.data.graph_limit} selection runs. Curves use the full selection history.`,
      );
    if (state.data.omitted_parents)
      warnings.push(
        `${state.data.omitted_parents} parent reference(s) are outside this view.`,
      );
    $("notice").hidden = !warnings.length;
    $("notice").textContent = warnings.join(" ");
    if (!state.data.nodes.some((run) => run.run_id === state.selected))
      state.selected = state.data.nodes.at(-1)?.run_id || null;
    if (!state.selected)
      $("inspector").replaceChildren(
        el("p", "RUN INSPECTOR", "eyebrow"),
        el("h3", "No selection runs"),
      );
    drawGraph();
    curves.scopes();
    runOptions();
    clearComparison();
  } catch (error) {
    $("notice").hidden = false;
    $("notice").textContent =
      `Refresh failed: ${error.message}. Any displayed data is from the previous successful read.`;
  } finally {
    $("refresh").disabled = false;
  }
}
$("refresh").addEventListener("click", refresh);
$("scope").addEventListener("change", curves.drawChart);
$("compare").addEventListener("click", compare);
for (const id of ["baseline", "candidate"]) {
  $(id).addEventListener("change", clearComparison);
}
refresh();
