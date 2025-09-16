
import os, io, json, re, requests
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# Optional: use RunPod serverless helper if available
try:
    import runpod
except Exception:
    runpod = None

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", None)

# dtype selection
if torch.cuda.is_available():
    dtype = torch.float16
else:
    dtype = torch.float32

# ---- Load model & processor once ----
processor = AutoProcessor.from_pretrained(
    MODEL_NAME, trust_remote_code=True, use_auth_token=HF_TOKEN
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    trust_remote_code=True,
    device_map="auto",
    use_auth_token=HF_TOKEN
)
model.eval()

def _load_image(inp):
    if "image_url" in inp and inp["image_url"]:
        r = requests.get(inp["image_url"], timeout=20)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    if "image_base64" in inp and inp["image_base64"]:
        b64 = inp["image_base64"]
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[-1]
        import base64
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise ValueError("Provide image_url or image_base64")

def _extract_json(text):
    m = re.search(r'\{.*\}', text, flags=re.S)
    if not m:
        raise ValueError("No JSON object in model output")
    return json.loads(m.group(0))

def predict(inp):
    prompt = inp.get("prompt") or "Return STRICT JSON with keys: caption, detected_objects, colors, nsfw. No prose."
    max_new_tokens = int(inp.get("max_new_tokens", 200))

    img = _load_image(inp)

    messages = [
        {"role":"system","content":"Return STRICT JSON with keys: caption, detected_objects, colors, nsfw. No prose, no markdown."},
        {"role":"user","content":[
            {"type":"image","image": img},
            {"type":"text","text": prompt}
        ]}
    ]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[img], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, do_sample=False, temperature=0.0, max_new_tokens=max_new_tokens)

    text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

    try:
        data = _extract_json(text)
    except Exception:
        data = {
            "caption": text.strip()[:300],
            "detected_objects": [],
            "colors": [],
            "nsfw": False
        }

    # ensure keys
    data.setdefault("caption","")
    data.setdefault("detected_objects",[])
    data.setdefault("colors",[])
    data.setdefault("nsfw",False)
    return {"output": data}

def handler(event):
    """RunPod serverless entrypoint. Expects { 'input': {...} }"""
    try:
        inp = event.get("input", {})
        return predict(inp)
    except Exception as e:
        return {"error": str(e)}

# Start serverless if runpod is present (on RunPod infra)
if runpod is not None:
    runpod.serverless.start({"handler": handler})
