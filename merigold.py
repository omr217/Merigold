import cv2
from PIL import Image
import torch
from tqdm import tqdm
import diffusers
import numpy as np 

device = "cuda"
path_out = "camera_output.gif"

pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
).to(device)
pipe.vae = diffusers.AutoencoderTiny.from_pretrained(
    "madebyollin/taesd", torch_dtype=torch.float16
).to(device)
pipe.set_progress_bar_config(disable=True)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera is not opened.")


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 10  

latent_common = torch.randn(
    (1, 4, 768 * height // (8 * max(width, height)), 768 * width // (8 * max(width, height)))
).to(device=device, dtype=torch.float16)

last_frame_latent = None
frames = []

print("Camera is on. To terminate press Ctrl+c or wait for the frames to be recorded.")

try:
    for _ in tqdm(range(50), desc="Frames are processing"):  
        ret, frame = cap.read()
        if not ret:
            break

    
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        latents = latent_common
        if last_frame_latent is not None:
            latents = 0.9 * latents + 0.1 * last_frame_latent

        depth = pipe(
            pil_image, match_input_resolution=False, latents=latents, output_latent=True
        )
        last_frame_latent = depth.latent
        depth_img = pipe.image_processor.visualize_depth(depth.prediction)[0]
        frames.append(depth_img)

finally:
    cap.release()


diffusers.utils.export_to_gif(frames, path_out, fps=fps)
print(f"Saved to {path_out} .")
