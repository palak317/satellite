# 🛰️ Satellite Imagery Reconstruction (GAN)
**Enhancing low-resolution satellite data and predicting missing landscape details using Generative Adversarial Networks.**

## 🌟 Project Overview
This project addresses the challenge of low-fidelity satellite data. Using a **Pix2Pix (Conditional GAN)** architecture, the application reconstructs blurry or low-resolution satellite images into high-definition landscape maps, "hallucinating" fine-grained details like building edges, road networks, and vegetation patterns.

### 🚀 Key Features
- **High-Fidelity Reconstruction:** Leverages U-Net generators to transform 256x256 low-res inputs into detailed outputs.
- **Interactive Dashboard:** Built with Streamlit for real-time inference and side-by-side visual comparison.
- **Geospatial Processing:** Handles satellite-specific data formats using OpenCV and PIL.
- **Edge Prediction:** Specifically tuned to recover sharp geometric features in urban and rural landscapes.

---

## 🛠️ Tech Stack
- **Deep Learning Framework:** [PyTorch](https://pytorch.org/)
- **Web Interface:** [Streamlit](https://streamlit.io/)
- **Architecture:** Pix2Pix (U-Net) / SRGAN
- **Image Processing:** OpenCV, PIL, NumPy
- **Deployment:** [Hugging Face Spaces](https://huggingface.co/spaces)

---

## 🏗️ Model Architecture
The core of this project is a **Conditional GAN (cGAN)**:
- **Generator:** A U-Net-based architecture with skip connections to preserve low-level spatial information from the input satellite feed.
- **Discriminator:** A PatchGAN classifier that penalizes structures at the scale of local image patches, ensuring realistic texture generation.
- **Loss Function:** A combination of Adversarial Loss and L1 Loss to ensure both visual realism and structural accuracy.

---

## 🏃 Installation & Setup

### 1. Clone & Environment
```bash
git clone [https://github.com/palak317/satellite-gan-reconstruction.git](https://github.com/palak317/satellite-gan-reconstruction.git)
cd satellite-gan-reconstruction
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
