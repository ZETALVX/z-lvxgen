/* ---------- Slider UI (guidance, strength, LoRA weight) ---------- */
const gSlider = document.getElementById("guidance"), gVal = document.getElementById("gval");
const sSlider = document.getElementById("strength"), sVal = document.getElementById("sval");
const lwSlider = document.getElementById("lora_w"), lwVal = document.getElementById("lwval");
[gSlider, sSlider, lwSlider].forEach(sl => {
  if (sl) sl.oninput = () => {
    const tgt = (sl === gSlider) ? gVal : (sl === sSlider ? sVal : lwVal);
    tgt.textContent = (+sl.value).toFixed(2);
  };
});

/* ---------- popola dropdown LoRA / checkpoint ---------- */
async function fillDropdown(url, selectEl) {
  const list = await (await fetch(url)).json();
  list.forEach(name => {
    const opt = document.createElement("option");
    opt.value = opt.textContent = name;
    selectEl.appendChild(opt);
  });
}
window.addEventListener("DOMContentLoaded", () => {
  fillDropdown("/api/loras",       document.getElementById("loraSelect"));
  fillDropdown("/api/checkpoints", document.getElementById("ckptSelect"));
});

/* ---------- Modal preview ---------- */
const modal = document.getElementById("modal"),
      modalImg = document.getElementById("modal-img");
document.getElementById("close-modal").onclick = () => modal.classList.add("hide");
modal.onclick = e => { if (e.target === modal) modal.classList.add("hide"); };

/* ---------- Canvas per inpaint ---------- */
const canvasBox = document.getElementById("canvas-box"),
      canvas    = document.getElementById("maskCanvas"),
      ctx       = canvas.getContext("2d"),
      modeSel   = document.getElementById("mode"),
      imgInput  = document.getElementById("image");
let drawing = false, brush = true;

function resizeCanvasToImage(img) {
  const size = 512;
  canvas.width = canvas.height = size;
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, size, size);
  ctx.globalAlpha = 0.5;
  ctx.drawImage(img, 0, 0, size, size);
  ctx.globalAlpha = 1.0;
}
function refreshCanvasPreview() {
  const file = imgInput.files[0];
  if (modeSel.value !== "inpaint" || !file) {
    canvasBox.classList.add("hide");
    return;
  }
  const img = new Image();
  img.onload = () => { canvasBox.classList.remove("hide"); resizeCanvasToImage(img); };
  img.src = URL.createObjectURL(file);
}
imgInput.onchange = refreshCanvasPreview;
modeSel.onchange  = refreshCanvasPreview;

/* disegno */
canvas.addEventListener("mousedown", () => drawing = true);
["mouseup","mouseleave"].forEach(ev => canvas.addEventListener(ev, () => { drawing = false; ctx.beginPath(); }));
canvas.addEventListener("mousemove", e => {
  if (!drawing) return;
  const r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  ctx.lineWidth = 20;
  ctx.lineCap = "round";
  ctx.strokeStyle = brush ? "white" : "black";
  ctx.lineTo(x,y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x,y);
});
document.getElementById("brushBtn").onclick = () => brush = true;
document.getElementById("eraserBtn").onclick = () => brush = false;
document.getElementById("clearBtn").onclick = () => { ctx.fillStyle="black"; ctx.fillRect(0,0,canvas.width,canvas.height); };

/* ---------- Submit form ---------- */
document.getElementById("gen-form").addEventListener("submit", async e => {
  e.preventDefault();
  const fd = new FormData();
  fd.append("engine",  document.getElementById("engine").value);
  fd.append("mode",    modeSel.value);
  fd.append("prompt",  document.getElementById("prompt").value);
  fd.append("negative_prompt", document.getElementById("neg").value);
  fd.append("strength", sSlider.value);
  fd.append("guidance", gSlider.value);
  fd.append("seed",     document.getElementById("seed").value);
  fd.append("steps",    document.getElementById("steps").value);
  fd.append("scheduler", document.getElementById("sched").value);
  fd.append("count",     document.getElementById("count").value);
  fd.append("lora_w",    lwSlider.value);
  fd.append("lora_name", document.getElementById("loraSelect").value);
  fd.append("ckpt_name", document.getElementById("ckptSelect").value);

  const initImg = imgInput.files[0];
  if (initImg) fd.append("image", initImg);
  if (modeSel.value === "inpaint") {
    const blob = await new Promise(res => canvas.toBlob(res, "image/png"));
    fd.append("mask", blob, "mask.png");
  }

  const res  = await fetch("/generate", { method: "POST", body: fd });
  const data = await res.json();
  if (data.images) {
    data.images.forEach(src => {
      const im = document.createElement("img");
      im.src = src;
      im.onclick = () => { modalImg.src = im.src; modal.classList.remove("hide"); };
      document.getElementById("gallery").prepend(im);
    });
  } else {
    alert(data.error || "Errore");
  }
});