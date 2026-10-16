// Grafikkortet: läser status från spelservern (/api/gpu) varje sekund och skickar den som
// händelsen "gpustatus". Gauges.render ritar ett instrument i bilstil i ett element.
(() => {
  const R = 46, CX = 60, CY = 60;  // bågen går 240° medurs, från nere till vänster till nere till höger
  const point = (degrees, radius) => {
    const a = degrees * Math.PI / 180;
    return [CX + radius * Math.cos(a), CY + radius * Math.sin(a)];
  };
  const [x0, y0] = point(150, R), [x1, y1] = point(30, R);
  const ARC = `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${R} ${R} 0 1 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
  const TICKS = Array.from({ length: 9 }, (_, i) => {
    const [ax, ay] = point(150 + i * 30, R + 9), [bx, by] = point(150 + i * 30, R + 13);
    return `<line class="g-tick" x1="${ax.toFixed(1)}" y1="${ay.toFixed(1)}" x2="${bx.toFixed(1)}" y2="${by.toFixed(1)}"/>`;
  }).join("");

  function build(el) {
    el.classList.add("gauge");
    el.innerHTML = `
      <svg viewBox="0 2 120 92" aria-hidden="true">${TICKS}
        <path class="g-track" d="${ARC}" pathLength="100"/>
        <path class="g-value" d="${ARC}" pathLength="100" stroke-dasharray="0 100"/>
      </svg>
      <div class="g-num"></div><div class="g-label"></div><div class="g-sub"></div>`;
    el.dataset.built = "1";
  }

  window.Gauges = {
    /** value/max avgör hur mycket av bågen som lyser; text är talet som visas (standard: value). */
    render(el, { value, max, text, unit = "", label, sub = "", tone }) {
      if (!el) return;
      if (!el.dataset.built) build(el);
      if (tone) el.style.setProperty("--tone", tone);
      const share = max && value != null ? Math.max(0, Math.min(1, value / max)) : 0;
      el.querySelector(".g-value").setAttribute("stroke-dasharray", `${(share * 100).toFixed(1)} 100`);
      const shown = text ?? (value == null ? "–" : String(value));
      el.querySelector(".g-num").innerHTML = `${shown}${unit && shown !== "–" ? `<small>${unit}</small>` : ""}`;
      el.querySelector(".g-label").textContent = label;
      el.querySelector(".g-sub").textContent = sub;
    },
  };

  async function poll() {
    if (document.visibilityState !== "visible") return;
    let status;
    try {
      status = await (await fetch("/api/gpu")).json();
    } catch {
      status = { error: "Start with python play.py to see the GPU." };
    }
    window.dispatchEvent(new CustomEvent("gpustatus", { detail: status }));
  }
  poll();
  setInterval(poll, 1000);
})();
