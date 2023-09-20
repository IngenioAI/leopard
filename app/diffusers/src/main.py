import torch
import json
from diffusers import StableDiffusionPipeline

with open("/data/input/params.json", "rt", encoding="utf-8") as fp:
    params = json.load(fp)

# make sure you're logged in with `huggingface-cli login`
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

if "prompt" in params:
    prompt = str(params["prompt"])
    image = pipe(prompt).images[0]
    image.save(f"/data/output/output.png")

    with open("/data/output/result.json", "wt", encoding="utf-8") as fp:
        json.dump({"image_path": "output.png"}, fp)
else:
    with open("/data/output/result.json", "wt", encoding="utf-8") as fp:
        json.dump({"success": False, "error_messsage": "prompt not found"}, fp)
