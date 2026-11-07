/* wassaw.io — small shared chart helpers (hand-built SVG, no libraries). */
const V = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const el = (t, a = {}, txt) => {
    const e = document.createElementNS(NS, t);
    for (const k in a) e.setAttribute(k, a[k]);
    if (txt != null) e.textContent = txt;
    return e;
  };
  // Linear scale with an inverse, so hover positions map back to data.
  const scale = (d0, d1, r0, r1) => {
    const f = v => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
    f.inv = p => d0 + (p - r0) / (r1 - r0) * (d1 - d0);
    return f;
  };
  const ticks = (lo, hi, n = 5) => {
    const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => s >= raw);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  };
  // Clears the svg and draws gridlines, y labels, x labels and axis titles.
  function frame(svg, o) {
    svg.textContent = "";
    o.yt.forEach(v => {
      svg.appendChild(el("line", { class: "gridline", x1: o.m.l, x2: o.W - o.m.r, y1: o.y(v), y2: o.y(v) }));
      svg.appendChild(el("text", { class: "axis-text", x: o.m.l - 8, y: o.y(v) + 4, "text-anchor": "end" },
        o.yfmt ? o.yfmt(v) : v));
    });
    (o.xt || []).forEach(t => svg.appendChild(el("text",
      { class: "axis-text", x: o.x(t.v), y: o.H - o.m.b + 18, "text-anchor": "middle" }, t.label)));
    if (o.xlab) svg.appendChild(el("text", { class: "axis-text", x: (o.m.l + o.W - o.m.r) / 2, y: o.H - 2, "text-anchor": "middle" }, o.xlab));
    if (o.ylab) svg.appendChild(el("text", { class: "axis-text", x: o.m.l - 8, y: o.m.t - 12 }, o.ylab));
  }
  const line = (pts, x, y) => pts.map((p, i) => (i ? "L" : "M") + x(p[0]).toFixed(1) + "," + y(p[1]).toFixed(1)).join(" ");
  const band = (xs, lo, hi, x, y) =>
    line(xs.map((v, i) => [v, hi[i]]), x, y) + " " +
    xs.map((v, i) => [v, lo[i]]).reverse().map(p => "L" + x(p[0]).toFixed(1) + "," + y(p[1]).toFixed(1)).join(" ") + " Z";
  // Tooltip box anchored at (x, y), flipped left when it would run off the right edge.
  function tip(svg, x, y, lines, W) {
    const w = Math.max(...lines.map(s => s.length)) * 6.7 + 18, h = lines.length * 16 + 10;
    const left = x + 12 + w > W ? x - 12 - w : x + 12;
    const g = el("g", { class: "tip", transform: `translate(${left.toFixed(1)},${(y - h / 2).toFixed(1)})` });
    g.appendChild(el("rect", { width: w, height: h, rx: 2 }));
    lines.forEach((s, i) => g.appendChild(el("text", { x: 9, y: 18 + i * 16 }, s)));
    svg.appendChild(g);
  }
  function table(t, head, rows) {
    t.innerHTML = "<tr>" + head.map(h => `<th>${h}</th>`).join("") + "</tr>" +
      rows.map(r => "<tr>" + r.map((c, i) => `<td${i ? ' class="num"' : ""}>${c}</td>`).join("") + "</tr>").join("");
  }
  function readout(div, items) {
    div.innerHTML = items.map(([k, v]) => `<div><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  }
  // Calls cb with the pointer's x in viewBox units, or null when the pointer leaves.
  function hover(svg, W, cb) {
    const px = e => { const r = svg.getBoundingClientRect(); return (e.clientX - r.left) / r.width * W; };
    svg.addEventListener("mousemove", e => cb(px(e)));
    svg.addEventListener("mouseleave", () => cb(null));
    svg.addEventListener("touchstart", e => cb(px(e.touches[0])), { passive: true });
    svg.addEventListener("touchmove", e => cb(px(e.touches[0])), { passive: true });
  }
  const usd = v => (v < 0 ? "−$" : "$") + Math.round(Math.abs(v)).toLocaleString("en-US");
  const pct = (v, d = 0) => (v * 100).toFixed(d) + "%";
  const ord = n => n + (n % 100 >= 11 && n % 100 <= 13 ? "th" : ["th", "st", "nd", "rd"][n % 10] || "th");
  // Standard normal, for pages that turn an interval into a decision.
  const N = {
    pdf: z => Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI),
    cdf: z => { const t = 1 / (1 + 0.2316419 * Math.abs(z));
      const p = 1 - N.pdf(z) * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
      return z >= 0 ? p : 1 - p; },
    inv: p => { let lo = -8, hi = 8; for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; N.cdf(m) < p ? lo = m : hi = m; } return (lo + hi) / 2; }
  };
  return { el, scale, ticks, frame, line, band, tip, table, readout, hover, usd, pct, ord, N };
})();
