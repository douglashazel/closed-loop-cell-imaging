/* Click-two-landmarks split-frame canvas for the Frame Shift tab. */

const ShiftCanvas = (() => {
  let canvas, ctx, img;
  let imgW = 0, imgH = 0;
  let leftWidth = 0;
  let points = []; // array of {x, y} in image coords (not canvas)
  let onShift = null;

  function init(canvasEl, onShiftChange) {
    canvas = canvasEl;
    ctx = canvas.getContext("2d");
    onShift = onShiftChange;

    canvas.addEventListener("click", (e) => {
      if (!img) return;
      const rect = canvas.getBoundingClientRect();
      const sx = (e.clientX - rect.left) / rect.width * imgW;
      const sy = (e.clientY - rect.top)  / rect.height * imgH;
      points.push({ x: sx, y: sy });
      render();
      if (points.length >= 2) {
        const p1 = points[points.length - 2];
        const p2 = points[points.length - 1];
        const dx = Math.round((p2.x - leftWidth) - p1.x);
        const dy = Math.round(p2.y - p1.y);
        if (onShift) onShift(dx, dy);
      }
    });
  }

  async function loadFrames(idx) {
    points = [];
    const url = `/api/frames/split?idx=${idx}&_t=${Date.now()}`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`failed: ${resp.status}`);
    leftWidth = parseInt(resp.headers.get("X-Left-Width") || "0", 10);
    const blob = await resp.blob();
    const url2 = URL.createObjectURL(blob);
    await new Promise((res, rej) => {
      const im = new Image();
      im.onload = () => { img = im; imgW = im.naturalWidth; imgH = im.naturalHeight; res(); };
      im.onerror = rej;
      im.src = url2;
    });
    // Set canvas dimensions to match aspect
    canvas.width  = imgW;
    canvas.height = imgH;
    render();
  }

  function setShift(dx, dy) {
    // Display as text overlay only — we re-render in render()
    render(dx, dy);
  }

  function clearPoints() {
    points = [];
    render();
  }

  function render(displayDx, displayDy) {
    if (!ctx) return;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (img) ctx.drawImage(img, 0, 0);

    // Divider
    ctx.strokeStyle = "rgba(255,255,255,0.3)";
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(leftWidth, 0);
    ctx.lineTo(leftWidth, canvas.height);
    ctx.stroke();
    ctx.setLineDash([]);

    // Points (last two highlighted)
    points.forEach((p, i) => {
      const isLast2 = i >= points.length - 2;
      ctx.fillStyle = isLast2 ? "#ff4a4a" : "rgba(255,120,120,0.5)";
      ctx.strokeStyle = "white";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });

    // Line between last two
    if (points.length >= 2) {
      const p1 = points[points.length - 2];
      const p2 = points[points.length - 1];
      ctx.strokeStyle = "rgba(255,74,74,0.6)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
  }

  return { init, loadFrames, clearPoints, setShift };
})();
