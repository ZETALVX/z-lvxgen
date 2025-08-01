#【Z】-【L】【v】【X】【G】【e】【n】
# LvXGen - By ZetaLvX
# ⚠️  DO NOT REMOVE THIS BANNER  ⚠️
# Keep the banner when reusing Z-LvXGen.

import os, gc, datetime, random, base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline, FluxPipeline,
    DPMSolverMultistepScheduler, EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler, UniPCMultistepScheduler
)

# ─────────── configuration ───────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE  = torch.float16 if DEVICE == 'cuda' else torch.float32

SDXL_BASE = os.getenv('SDXL_BASE',
                      '/path/to/stabilityai-stable-diffusion-xl-base-1-0')
FLUX_BASE      = os.getenv('FLUX_BASE',      '/path/to/FLUX.1-schnell')
FLUX_CTX_BASE  = os.getenv('FLUX_CTX_BASE',  '/path/to/FLUX.1-Kontext-dev')

# cartelle LoRA / checkpoint centralizzate
LORA_DIR = os.path.join('loras')
CKPT_DIR = os.path.join('checkpoints')
os.makedirs(LORA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

UPLOAD_DIR = os.path.join('static', 'images')
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024   # 1 GB

# ─────────── schedulers for SDXL ───────────
SCHEDULERS = {
    'DPMPP2M': lambda cfg: DPMSolverMultistepScheduler.from_config(
        cfg, algorithm_type='dpmsolver++', use_karras_sigmas=True),
    'Euler':  lambda cfg: EulerDiscreteScheduler.from_config(cfg),
    'EulerA': lambda cfg: EulerAncestralDiscreteScheduler.from_config(cfg),
    'UniPC':  lambda cfg: UniPCMultistepScheduler.from_config(cfg),
}

# ─────────── lazy-loaded pipelines ───────────
_pipelines = {}
def _load_sdxl_pipeline(mode: str, checkpoint_path: str | None = None):
    key = ('sdxl', mode, checkpoint_path or '')
    if key in _pipelines:
        return _pipelines[key]

    if checkpoint_path:
        pipe = StableDiffusionXLPipeline.from_single_file(
            checkpoint_path, torch_dtype=DTYPE, safety_checker=None)
    else:
        if mode == 't2i':
            pipe = StableDiffusionXLPipeline.from_pretrained(
                SDXL_BASE, torch_dtype=DTYPE, variant='fp16', use_safetensors=True)
        elif mode == 'img2img':
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                SDXL_BASE, torch_dtype=DTYPE, variant='fp16', use_safetensors=True)
        else:  # inpaint
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                SDXL_BASE, torch_dtype=DTYPE, variant='fp16', use_safetensors=True)

    if DEVICE == 'cuda':
        pipe.to(DEVICE)
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()
    _pipelines[key] = pipe
    return pipe

def _load_flux_pipeline(base_path: str):
    key = ('flux', base_path)
    if key in _pipelines:
        return _pipelines[key]
    pipe = FluxPipeline.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16 if DEVICE == 'cuda' else torch.float32)
    if DEVICE == 'cuda':
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
    _pipelines[key] = pipe
    return pipe

def get_pipeline(engine: str, mode: str, checkpoint_path: str | None = None):
    if engine == 'sdxl':
        return _load_sdxl_pipeline(mode, checkpoint_path)
    elif engine == 'flux':
        return _load_flux_pipeline(FLUX_BASE)
    elif engine == 'flux_ctx':
        return _load_flux_pipeline(FLUX_CTX_BASE)
    else:
        raise ValueError('Unknown engine')

# ─────────── LoRA helper ───────────
def apply_lora(pipe, lora_path: str, weight: float = .7):
    lora_dir, fname = os.path.split(lora_path)
    adapter_name = os.path.splitext(fname)[0]
    if not hasattr(pipe, '_loaded_loras'):
        pipe._loaded_loras = set()
    if adapter_name not in pipe._loaded_loras:
        pipe.load_lora_weights(lora_dir, weight_name=fname, adapter_name=adapter_name)
        pipe._loaded_loras.add(adapter_name)
    pipe.set_adapters([adapter_name], adapter_weights=[weight])

# ─────────── helpers ───────────
def list_safetensors(folder):
    return sorted(f for f in os.listdir(folder)
                  if f.endswith('.safetensors') and os.path.isfile(os.path.join(folder, f)))

# ─────────── routes ───────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/loras')
def api_loras():
    return jsonify(list_safetensors(LORA_DIR))

@app.route('/api/checkpoints')
def api_checkpoints():
    return jsonify(list_safetensors(CKPT_DIR))

@app.route('/generate', methods=['POST'])
def generate():
    form  = request.form
    files = request.files  # serve solo per l’immagine/mask

    engine = form.get('engine', 'sdxl')
    mode   = form.get('mode', 't2i')
    prompt = (form.get('prompt') or '').strip()
    if not prompt:
        return jsonify({'error': 'Prompt required'}), 400
    neg = form.get('negative_prompt') or None

    strength = float(form.get('strength', '0.6'))
    guidance = float(form.get('guidance', '7.5'))
    steps    = int(form.get('steps', '30'))
    seed_val = int(form.get('seed', '-1'))
    count    = max(1, min(int(form.get('count', '1')), 8))
    scheduler = form.get('scheduler', 'DPMPP2M')

    lora_weight = float(form.get('lora_w', '0.7'))
    lora_name   = form.get('lora_name', '')
    ckpt_name   = form.get('ckpt_name', '')

    lora_path = os.path.join(LORA_DIR, lora_name) if lora_name else None
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name) if ckpt_name else None
    if lora_path and not os.path.isfile(lora_path):
        return jsonify({'error': 'LoRA not found'}), 400
    if ckpt_path and not os.path.isfile(ckpt_path):
        return jsonify({'error': 'Checkpoint not found'}), 400

    init_img_file = files.get('image')
    mask_file     = files.get('mask')

    if mode != 't2i' and not (init_img_file and init_img_file.filename):
        return jsonify({'error': 'Initial image required'}), 400
    if mode == 'inpaint' and not (mask_file and mask_file.filename):
        return jsonify({'error': 'Mask required'}), 400

    pipe = get_pipeline(engine, mode, ckpt_path)

    if engine == 'sdxl' and scheduler in SCHEDULERS:
        pipe.scheduler = SCHEDULERS[scheduler](pipe.scheduler.config)
    if lora_path and engine == 'sdxl':
        apply_lora(pipe, lora_path, lora_weight)

    # RNG
    g = torch.Generator(device=DEVICE)
    if seed_val == -1:
        seed_val = random.randint(0, 2**32 - 1)
    g.manual_seed(seed_val)

    images_data = []
    for i in range(count):
        gen_seed = (seed_val + i) & 0xFFFFFFFF
        g.manual_seed(gen_seed)

        if engine == 'sdxl':
            if mode == 't2i':
                img = pipe(prompt=prompt, negative_prompt=neg,
                           guidance_scale=guidance,
                           num_inference_steps=steps,
                           generator=g).images[0]
            elif mode == 'img2img':
                init = Image.open(init_img_file).convert('RGB').resize((1024, 1024))
                img = pipe(prompt=prompt, negative_prompt=neg,
                           image=init, strength=strength,
                           guidance_scale=guidance,
                           num_inference_steps=steps,
                           generator=g).images[0]
            else:  # inpaint
                init = Image.open(init_img_file).convert('RGB').resize((1024, 1024))
                mask = Image.open(mask_file).convert('L').resize((1024, 1024))
                img = pipe(prompt=prompt, negative_prompt=neg,
                           image=init, mask_image=mask,
                           strength=strength,
                           guidance_scale=guidance,
                           num_inference_steps=steps,
                           generator=g).images[0]
        else:  # flux / flux_ctx
            img = pipe(prompt=prompt,
                       guidance_scale=0.0,
                       num_inference_steps=4,
                       max_sequence_length=256,
                       generator=torch.Generator(device='cpu').manual_seed(gen_seed)
                       ).images[0]

        # ─── salvataggio PNG su disco ───
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = (
            f"{timestamp}_seed-{gen_seed}"
            f"_{engine}-{mode}"
            f"_steps-{steps}"
            f"_guid-{str(guidance).replace('.', '_')}"
            + (f"_str-{str(strength).replace('.', '_')}" if mode != 't2i' else '')
            + ".png"
        )
        img.save(os.path.join(UPLOAD_DIR, fname))

        # ─── anteprima base64 per la UI (come prima) ───
        buf = BytesIO()
        img.save(buf, format='PNG')
        images_data.append('data:image/png;base64,' +
                           base64.b64encode(buf.getvalue()).decode())

        del img
        torch.cuda.empty_cache()
        gc.collect()

    return jsonify({
        'images': images_data,
        'seed': seed_val,
        'count': count,
        'engine': engine,
        'mode': mode
    })

@app.route('/static/images/<path:fname>')
def serve_img(fname):
    return send_from_directory(UPLOAD_DIR, fname)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8443, ssl_context='adhoc', threaded=True)
