# DCGAN Project

This is an implementation of a Deep Convolutional GAN (DCGAN) using PyTorch/TensorFlow and Django for the web interface.
It uses **PyTorch** and trained on the CelebA dataset.

## Project Structure
- `DCGAN.py` : Main GAN training script
- `ImageApp/` : Django app
- `tools/` : Utility scripts
- `output/` : Generated images (ignored in git)
- `Dataset/`, `CelebDataset/` : Training datasets (ignored in git)

## 🚀 Features
- DCGAN architecture (Generator + Discriminator)
- Training on CelebA dataset
- Generates realistic human face images
- Organized output pipeline


## Setup
```bash
git clone https://github.com/Aditya1407g/human-face-DCGAN.git
cd your-repo-name
pip install -r requirements.txt
