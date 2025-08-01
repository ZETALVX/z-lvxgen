# Z-LvXgen — Local SDXL / FLUX WebUI

Self-hosted image‑generation suite powered by **Stable Diffusion XL** and **FLUX 1** (base & Kontext).  
Runs entirely on your own machine (GPU **or** CPU) and exposes a clean browser UI for:

* Text-to-Image (SDXL & FLUX)
* **Img2Img** (SDXL)
* **Inpaint** with live mask-painting (SDXL)
* Centralised **LoRA** & **checkpoint** selection
* Multiple schedulers (DPM++ 2M Karras, Euler, Euler a, UniPC)
* HTTPS by default – no login, no telemetry

Tutorial: https://youtu.be/Fjg5idRR44o  
Donations: https://ko-fi.com/zetalvx
---

## ✨ Key features

| Feature | SDXL | FLUX 1 | FLUX 1 Kontext |
|---------|------|--------|----------------|
| Text-to-Image | ✅ | ✅ | ✅ (long‑context) |
| Img2Img | ✅ | – | – |
| Inpaint | ✅ | – | – |
| LoRA | ✅ | – | – |
| Custom checkpoint | ✅ | – | – |

> **FLUX Kontext** keeps up to ~4 000 tokens of prompt context – perfect for storyboards and multi-shot scenes.

---

## 🚀 Quick‑start

```bash
git clone https://github.com/zetalvx/Z-LvXgen.git
cd Z-LvXgen
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # diffusers, torch, flask…

# OPTIONAL – GPU speed‑ups
pip install flash-attn==2.6.3 sageattention==2.1.1

# Point to your models
export SDXL_BASE=/path/to/stable-diffusion-xl-base-1-0
export FLUX_BASE=/path/to/FLUX.1-schnell
export FLUX_CTX_BASE=/path/to/FLUX.1-Kontext-dev

python app.py         # HTTPS at https://localhost:8443
```

Open your browser, accept the self‑signed cert, and start creating!

---

## 📂 Directory layout

```
Z-LvXgen/
├─ app.py
├─ requirements.txt
├─ templates/index.html
├─ static/
│  ├─ css/style.css
│  ├─ js/script.js
│  └─ images/            ← generated previews
├─ loras/                ← drop *.safetensors LoRA files
├─ checkpoints/          ← drop SDXL checkpoints
├─ LICENSE
└─ NOTICE
```

The **LoRA** and **checkpoint** dropdowns are auto‑populated from those two folders.

---

## 🏃‍ Running

```bash
# HTTPS (default)
python app.py

# Plain HTTP (omit ssl_context)
python -c "import app; app.app.run(host='0.0.0.0', port=5000, threaded=True)"
```

---

## 🌐 Web UI walkthrough

1. **Engine** → SDXL / FLUX / FLUX Context  
2. **Mode** → Text-to-Image, Img2Img, or Inpaint  
3. **Prompt** → describe your scene  
4. *(Inpaint)* upload image → paint mask (white = change)  
5. Optionally choose LoRA or checkpoint  
6. Fine-tune params (Guidance, Steps, Seed, Scheduler, Count)  
7. **Generate** → images pop into the gallery (click to zoom)

---

## 🖌️ Inpaint tips

* Strength 0.4-0.8 typical  
* White brush = regenerate, Black = keep  
* Higher guidance makes edits follow the prompt more strictly  

---

## ⚡ Troubleshooting

| Issue | Fix |
|-------|-----|
| `flash-attn` / `sageattention` warnings | Optional; install for GPU speed-ups |
| `cryptography` missing | `pip install cryptography` or use HTTP |
| CUDA OOM | Lower Steps / Count or image resolution |

---

## 📦 Adding new LoRA / checkpoints

1. Copy `.safetensors` into `loras/` or `checkpoints/`  
2. Refresh browser – files appear in dropdowns

---

## License & attribution

Z‑LvXgen is released under the **Apache 2.0** license (see `LICENSE`).

It embeds:

| Model | License |
|-------|---------|
| Stable Diffusion XL 1.0 | CreativeML Open RAIL++ M |
| FLUX 1 & FLUX 1 Kontext | Apache 2.0 - Flux-1-dev-non-commercial-license

When you reuse or fork this repository **you must**:

1. Keep the credit **“By ZetaLvX”** in both code banner and any UI.  
2. Preserve the `LICENSE` and `NOTICE` files, and this section.  
3. Comply with the upstream licenses listed above.

---

## 🙏 Acknowledgements

* Stability AI – Stable Diffusion XL  
* Lattice Labs – FLUX 1 / Kontext
* Hugging Face Diffusers  
