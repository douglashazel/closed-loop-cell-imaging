/* Draggable circle ROI overlaid on a frame image. */

const RoiCanvas = (() => {
  let wrap, svg, circle, handle, img;
  let imgNatW = 0, imgNatH = 0;
  let state = { radius: 2000, cx: 0, cy: 0 };  // in image coords
  let onChange = null;
  let dragging = null; // "move" | "resize" | null

  function init(wrapEl, imgEl, circleEl, handleEl, onChangeCb) {
    wrap = wrapEl;
    img = imgEl;
    svg = wrap.querySelector("svg");
    circle = circleEl;
    handle = handleEl;
    onChange = onChangeCb;

    circle.addEventListener("pointerdown", (e) => startDrag(e, "move"));
    handle.addEventListener("pointerdown", (e) => startDrag(e, "resize"));
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", endDrag);
  }

  function setImage(imgW, imgH) {
    imgNatW = imgW;
    imgNatH = imgH;
    svg.setAttribute("viewBox", `0 0 ${imgW} ${imgH}`);
    render();
  }

  function setState(radius, xShift, yShift) {
    if (!imgNatW || !imgNatH) return;
    state.radius = radius;
    state.cx = imgNatW / 2 + xShift;
    state.cy = imgNatH / 2 + yShift;
    render();
  }

  function render() {
    if (!imgNatW) return;
    circle.setAttribute("cx", state.cx);
    circle.setAttribute("cy", state.cy);
    circle.setAttribute("r",  state.radius);
    handle.setAttribute("cx", state.cx + state.radius);
    handle.setAttribute("cy", state.cy);
    handle.setAttribute("r",  Math.max(8, Math.min(40, imgNatW / 100)));
    circle.setAttribute("stroke-width", Math.max(2, imgNatW / 400));
  }

  function clientToImage(e) {
    const rect = wrap.getBoundingClientRect();
    const xNorm = (e.clientX - rect.left) / rect.width;
    const yNorm = (e.clientY - rect.top)  / rect.height;
    return { x: xNorm * imgNatW, y: yNorm * imgNatH };
  }

  function startDrag(e, mode) {
    e.preventDefault();
    dragging = mode;
  }

  function onMove(e) {
    if (!dragging) return;
    const { x, y } = clientToImage(e);
    if (dragging === "move") {
      state.cx = x;
      state.cy = y;
    } else if (dragging === "resize") {
      const dx = x - state.cx;
      const dy = y - state.cy;
      state.radius = Math.max(10, Math.round(Math.hypot(dx, dy)));
    }
    render();
    fireChange();
  }

  function endDrag() { dragging = null; }

  let changeTimer = null;
  function fireChange() {
    if (changeTimer) clearTimeout(changeTimer);
    changeTimer = setTimeout(() => {
      if (!onChange) return;
      const xShift = Math.round(state.cx - imgNatW / 2);
      const yShift = Math.round(state.cy - imgNatH / 2);
      onChange({ radius: Math.round(state.radius), x_shift: xShift, y_shift: yShift });
    }, 100);
  }

  return { init, setImage, setState };
})();
