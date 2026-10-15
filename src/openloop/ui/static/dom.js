export const $ = (id) => document.getElementById(id);
export const palette = [
  "#148774",
  "#6c79b5",
  "#c68c41",
  "#bd6d82",
  "#6098ad",
  "#89934d",
];
export const short = (value) => String(value).slice(0, 8);
export function fmt(value) {
  if (value == null) return "—";
  const number = Number(value);
  if (number !== 0 && Math.abs(number) < 0.0001) return number.toExponential(3);
  return number.toLocaleString(undefined, { maximumFractionDigits: 5 });
}
export function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}
export function svg(tag, attrs = {}, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs))
    node.setAttribute(key, String(value));
  if (text != null) node.textContent = String(text);
  return node;
}
export async function api(path) {
  const response = await fetch(path, { cache: "no-store" });
  let data = null;
  try {
    data = await response.json();
  } catch {
    // A failed request may not return JSON; report its status instead.
  }
  if (!response.ok || data === null)
    throw new Error(data?.error || `Cannot load evidence (HTTP ${response.status})`);
  return data;
}
export function table(headers, rows) {
  const node = el("table"),
    head = el("tr"),
    body = el("tbody");
  headers.forEach((header) => head.append(el("th", header)));
  const thead = el("thead");
  thead.append(head);
  node.append(thead);
  for (const row of rows) {
    const tr = el("tr");
    row.forEach((cell) => {
      const td = el("td");
      td.append(cell instanceof Node ? cell : el("span", cell));
      tr.append(td);
    });
    body.append(tr);
  }
  node.append(body);
  return node;
}
