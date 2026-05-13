# Dual-Branch Semantic and Frequency Analysis for Detection of AI-Generated Face Images

A dual-branch deep learning framework for detecting AI-generated face images, combining a **semantic branch** (CLIP ViT-L/14) with a **frequency-domain branch** (EfficientNet-B0 on DCT-transformed inputs), fused through a late-fusion meta-classifier.

## Overview

Modern generative models (Stable Diffusion, FLUX, Z-Image, etc.) produce face images that are increasingly difficult to distinguish from real photographs using either semantic features or frequency artefacts alone. This project investigates whether combining both views — high-level semantic representations from a vision-language model and low-level frequency signatures from the Discrete Cosine Transform — yields a more robust detector than either branch in isolation.

## Architecture

The pipeline consists of three components:

- **Semantic branch** — frozen CLIP ViT-L/14 image encoder producing 768-dimensional embeddings, capturing high-level perceptual cues.
- **Frequency branch** — EfficientNet-B0 trained on log-magnitude DCT representations of the input image, capturing frequency-domain fingerprints left by generative models.
- **Late-fusion meta-classifier** — a lightweight head that takes the calibrated probability outputs of both branches and learns the optimal decision boundary.

## Dataset

A custom 6,000-image dataset of real and AI-generated face images was built for this study, balanced across multiple generators to encourage cross-generator generalisation. The dataset is published on HuggingFace:

- [`Madushan996/real_fake_images`](https://huggingface.co/datasets/Madushan996/real_fake_images)

## Results

On a held-out test split, the late-fusion model achieves:

| Metric | Value |
|---|---|
| Test accuracy | **99.78%** |
| False negatives | **0** |
| Branches | CLIP ViT-L/14 + EfficientNet-B0 (DCT) |

The fused model consistently outperforms either branch evaluated independently, supporting the central hypothesis that semantic and frequency cues are complementary.

## Repository Contents

| Notebook | Purpose |
|---|---|
| `Notebook_1_Data_Preprocessing.ipynb` | Dataset assembly, face cropping, train/val/test splits, DCT preparation |
| `Notebook_2_Model_Training_1.ipynb` | Training both branches and the late-fusion meta-classifier |
| `Notebook_3_Evaluation_Demo.ipynb` | Quantitative evaluation, confusion matrices, and ablations |
| `Deepfake_Detector_Inference_FaceDetect.ipynb` | End-to-end inference with face detection on arbitrary input images |

## Reproduction

The notebooks were developed on Google Colab and Kaggle (T4 GPU). To reproduce:

1. Run `Notebook_1_Data_Preprocessing.ipynb` to prepare the dataset (or pull directly from HuggingFace).
2. Run `Notebook_2_Model_Training_1.ipynb` to train the two branches and the fusion head.
3. Run `Notebook_3_Evaluation_Demo.ipynb` for evaluation metrics.
4. Use `Deepfake_Detector_Inference_FaceDetect.ipynb` to test on new images.

Main dependencies: `torch`, `transformers`, `timm`, `scikit-learn`, `opencv-python`, `numpy`, `pandas`, `matplotlib`.

## Acknowledgements

This work was developed as part of the MSc Artificial Intelligence programme at Anglia Ruskin University (via CINEC Campus, Sri Lanka). Pretrained weights for CLIP ViT-L/14 are from OpenAI; EfficientNet-B0 weights are from `timm`.

## Author

**Madushan Bhashana Dissanayake** — [github.com/Madushan996](https://github.com/Madushan996)
