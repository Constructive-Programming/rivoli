// Right-hand nav icon back to the brand site, placed before mdBook's GitHub icon. This
// matches the order on the other constructive.dev sites.
(() => {
  const bar = document.querySelector(".right-buttons");
  if (!bar) return;
  const a = document.createElement("a");
  a.href = "https://www.constructive.dev/";
  a.title = "constructive.dev";
  a.setAttribute("aria-label", "constructive.dev");
  a.className = "rv-brand-link";
  a.innerHTML = '<img src="https://www.constructive.dev/assets/img/favicon.svg" alt="">';
  bar.prepend(a);
})();

// Click-to-sort table columns. A column sorts numerically when every cell is a number
// (with an optional B/T/M suffix, first number of "753B / 40B" wins) or a missing marker
// (— or N/A); otherwise it sorts as text. Missing values stay last in both directions.
// Footnote markers are ignored, so "3.0⁶" reads as 3.0.
(() => {
  const SCALE = { T: 1000, B: 1, M: 0.001 };
  const missing = (s) => s === "" || s === "—" || s === "N/A";
  const cellText = (td) => {
    const c = td.cloneNode(true);
    c.querySelectorAll("sup").forEach((s) => s.remove());
    return c.textContent.trim();
  };
  const num = (s) => {
    const m = s.match(/^(\d+(?:\.\d+)?)\s*([TBM])?/);
    return m ? parseFloat(m[1]) * (SCALE[m[2]] ?? 1) : null;
  };
  document.querySelectorAll(".content main table").forEach((table) => {
    const body = table.tBodies[0];
    if (!body) return;
    table.querySelectorAll("thead th").forEach((th, col) => {
      const btn = document.createElement("button");
      btn.className = "rv-sort";
      btn.append(...th.childNodes);
      th.append(btn);
      btn.addEventListener("click", () => {
        const rows = [...body.rows];
        const texts = rows.map((r) => cellText(r.cells[col]));
        const numeric = texts.every((t) => missing(t) || num(t) !== null);
        const dir = th.getAttribute("aria-sort") === "ascending" ? -1 : 1;
        table.querySelectorAll("thead th").forEach((h) => h.removeAttribute("aria-sort"));
        th.setAttribute("aria-sort", dir === 1 ? "ascending" : "descending");
        const key = (i) => (missing(texts[i]) ? null : numeric ? num(texts[i]) : texts[i].toLowerCase());
        const order = rows.map((_, i) => i).sort((a, b) => {
          const ka = key(a), kb = key(b);
          if (ka === null || kb === null) return (ka === null) - (kb === null);
          return (ka < kb ? -1 : ka > kb ? 1 : 0) * dir;
        });
        order.forEach((i) => body.append(rows[i]));
      });
    });
  });
})();
