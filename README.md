# 📸 Real-Time Depth Estimation with Marigold and Webcam

This project captures live webcam video and generates corresponding **depth map visualizations** using the [Marigold Depth LCM model](https://huggingface.co/prs-eth/marigold-depth-lcm-v1-0). The output is saved as an animated `.gif`, making it easy to observe temporal depth changes in a short sequence.

---

## 🔧 Features

- ✅ Real-time frame capture from webcam
- 🧠 Depth estimation with [Marigold LCM](https://huggingface.co/prs-eth/marigold-depth-lcm-v1-0) (Latent Consistency Model)
- 🌀 Temporal consistency via latent blending across frames
- ⚡ Optimized for GPU (`cuda`) with `torch.float16` precision
- 💾 Saves the output as an animated GIF (`camera_output.gif`)

---

## 📦 Requirements

- Python 3.8+
- A webcam
- GPU with CUDA support (recommended)

---

### 📄 License
This project uses models available under respective licenses from Hugging Face. Refer to their pages for terms of use.

---

#### 🙌 Acknowledgements
•	Diffusers by Hugging Face
•	Marigold LCM
•	Autoencoder Tiny (taesd)

