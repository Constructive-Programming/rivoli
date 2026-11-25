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
