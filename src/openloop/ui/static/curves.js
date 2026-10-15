import { $, palette, short, fmt, el, svg, table } from "/dom.js";

// One series per candidate and fidelity; lines never join fidelities.
const seriesKey = (series) => `${series.candidate_hash} ${series.fidelity}`;
const seriesLabel = (series) =>
  `${short(series.candidate_hash)} · ${series.fidelity}`;

export function curveView(state, selectRun) {
  function scopes() {
    const select = $("scope"),
      previous = select.value,
      seen = new Set();
    select.replaceChildren();
    for (const series of state.data.series) {
      if (seen.has(series.scope_hash)) continue;
      seen.add(series.scope_hash);
      const scope = series.scope,
        mode = scope.inputs.execution.simulated === true ? " · simulated" : "";
      const option = el(
        "option",
        `${scope.metric} · ${scope.inputs.phase} · ${short(series.scope_hash)}${mode}`,
      );
      option.value = series.scope_hash;
      select.append(option);
    }
    if (seen.has(previous)) select.value = previous;
    select.disabled = !seen.size;
    $("point-details").hidden = !seen.size;
    drawChart();
  }
  function drawChart() {
    const holder = $("chart"),
      legend = $("series-legend"),
      hash = $("scope").value;
    holder.replaceChildren();
    legend.replaceChildren();
    const all = state.data.series.filter(
      (series) => series.scope_hash === hash,
    );
    $("scope-details").textContent = all.length
      ? `Data ${short(all[0].scope.inputs.data_hash)} · environment ${short(all[0].scope.inputs.environment_hash)} · ${all[0].scope.direction}`
      : "";
    if (!all.length) {
      holder.append(
        el("p", "No completed validation metrics to plot.", "empty"),
      );
      $("point-evidence").replaceChildren();
      return;
    }
    const active = all.filter((series) => !state.hidden.has(seriesKey(series)));
    all.forEach((series, index) => {
      const button = el("button", null),
        dot = svg("svg", { viewBox: "0 0 10 10" });
      dot.append(
        svg("circle", {
          cx: 5,
          cy: 5,
          r: 4,
          fill: palette[index % palette.length],
        }),
      );
      const key = seriesKey(series);
      button.append(dot, el("span", seriesLabel(series)));
      button.setAttribute("aria-pressed", String(!state.hidden.has(key)));
      button.title = `${series.candidate_hash} · ${series.fidelity}`;
      button.addEventListener("click", () => {
        if (state.hidden.has(key)) state.hidden.delete(key);
        else state.hidden.add(key);
        drawChart();
      });
      legend.append(button);
    });
    const evidence = [];
    all.forEach((series) =>
      series.points.forEach((point) => {
        const link = el(
          "button",
          `${point.run_ids.length} run${point.run_ids.length === 1 ? "" : "s"}`,
        );
        link.addEventListener("click", () => {
          const node = point.run_ids.find((id) =>
            state.data.nodes.some((run) => run.run_id === id),
          );
          if (node) {
            selectRun(node);
            $("lineage").scrollIntoView();
          } else {
            $("notice").hidden = false;
            $("notice").textContent =
              "Point evidence is older than the graph window. Full run IDs are listed below.";
          }
        });
        const details = el("details");
        details.append(
          el("summary", "Run IDs"),
          el("pre", point.run_ids.join("\n")),
        );
        const refs = el("div");
        refs.append(link, details);
        evidence.push([
          seriesLabel(series),
          fmt(point.budget),
          fmt(point.mean),
          fmt(point.sd),
          `${point.seed_count} seeds / ${point.attempt_count} attempts`,
          refs,
        ]);
      }),
    );
    $("point-evidence").replaceChildren(
      table(
        [
          "Candidate · fidelity",
          "Budget",
          "Mean",
          "Seed SD",
          "Evidence",
          "Runs",
        ],
        evidence,
      ),
    );
    if (!active.length) {
      holder.append(
        el("p", "Select a candidate in the legend to show its curve.", "empty"),
      );
      return;
    }
    const points = active.flatMap((series) => series.points),
      values = points.flatMap((point) => [
        point.low ?? point.mean,
        point.high ?? point.mean,
      ]);
    let minX = Math.min(...points.map((p) => p.budget)),
      maxX = Math.max(...points.map((p) => p.budget));
    let minY = Math.min(...values),
      maxY = Math.max(...values);
    const pad = (maxY - minY || Math.abs(minY) * 0.08 || 1) * 0.15;
    minY -= pad;
    maxY += pad;
    if (minX === maxX) {
      minX = Math.max(0, minX * 0.8);
      maxX = maxX * 1.2 || 1;
    }
    const x = (value) => 65 + ((value - minX) / (maxX - minX)) * 760,
      y = (value) => 245 - ((value - minY) / (maxY - minY)) * 215;
    const drawing = svg("svg", {
      viewBox: "0 0 860 300",
      role: "group",
      "aria-label": `${all[0].scope.metric}, mean with one standard deviation between seeds`,
    });
    for (let index = 0; index < 5; index++) {
      const value = minY + ((maxY - minY) * index) / 4;
      drawing.append(
        svg("line", {
          x1: 65,
          x2: 825,
          y1: y(value),
          y2: y(value),
          class: "gridline",
        }),
        svg(
          "text",
          { x: 53, y: y(value) + 3, "text-anchor": "end" },
          fmt(value),
        ),
      );
    }
    for (let index = 0; index < 4; index++) {
      const value = minX + ((maxX - minX) * index) / 3;
      drawing.append(
        svg(
          "text",
          { x: x(value), y: 269, "text-anchor": "middle" },
          fmt(value),
        ),
      );
    }
    drawing.append(
      svg(
        "text",
        { x: 445, y: 293, "text-anchor": "middle" },
        `Requested work · ${all[0].scope.inputs.budget_unit}`,
      ),
      svg(
        "text",
        { x: 65, y: 15 },
        `${all[0].scope.metric} · ${all[0].scope.unit}`,
      ),
    );
    active.forEach((series) => {
      const color = palette[all.indexOf(series) % palette.length],
        coords = series.points
          .map((point) => `${x(point.budget)},${y(point.mean)}`)
          .join(" ");
      // Never bridge a band across a point without a measured seed SD.
      let segment = [];
      function band() {
        if (segment.length > 1)
          drawing.append(
            svg("polygon", {
              points: [
                ...segment.map((p) => `${x(p.budget)},${y(p.low)}`),
                ...segment
                  .toReversed()
                  .map((p) => `${x(p.budget)},${y(p.high)}`),
              ].join(" "),
              fill: color,
              class: "band",
            }),
          );
        segment = [];
      }
      for (const point of series.points) {
        if (point.sd == null) band();
        else segment.push(point);
      }
      band();
      drawing.append(
        svg("polyline", { points: coords, stroke: color, class: "curve-line" }),
      );
      series.points.forEach((point) => {
        if (point.sd != null)
          drawing.append(
            svg("line", {
              x1: x(point.budget),
              x2: x(point.budget),
              y1: y(point.low),
              y2: y(point.high),
              stroke: color,
              "stroke-width": 2,
            }),
          );
        const mark = svg("circle", {
          cx: x(point.budget),
          cy: y(point.mean),
          r: 4.5,
          fill: color,
          class: "point",
          tabindex: 0,
          role: "button",
          "aria-label": `${seriesLabel(series)} at ${point.budget}, mean ${fmt(point.mean)}`,
        });
        mark.append(
          svg(
            "title",
            {},
            `${seriesLabel(series)}\nMean ${fmt(point.mean)}, SD ${fmt(point.sd)}\n${point.seed_count} seeds, ${point.attempt_count} attempts`,
          ),
        );
        const showEvidence = () => {
          $("point-details").open = true;
          $("point-details").scrollIntoView({ block: "nearest" });
        };
        mark.addEventListener("click", showEvidence);
        mark.addEventListener("keydown", (event) => {
          if (["Enter", " "].includes(event.key)) {
            event.preventDefault();
            showEvidence();
          }
        });
        drawing.append(mark);
      });
    });
    holder.append(drawing);
  }

  return { scopes, drawChart };
}
