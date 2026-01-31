# Image2Shape

This repository contains the implementation for our project in the **Advanced Deep Learning for Computer Vision (ADL4CV)** course at the **Technical University of Munich (TUM)**.

## Installation

Since this project builds on both **ImageDream Diffusion** and **SnapGTR**, please follow the installation instructions in their respective repositories first and create two separate conda environments:

* **ImageDream Environment**: Follow the setup at [ImageDream (ByteDance)](https://github.com/bytedance/ImageDream/tree/main/extern/ImageDream)
* **SnapGTR Environment**: Follow the setup at [SnapGTR (Snap Research)](https://github.com/snap-research/snap_gtr/)

## Checkpoints

Donwload the **SnapGTR** checkpoint from here: [SnapGTR Checkpoint](https://huggingface.co/snap-research/gtr/blob/main/ckpt.pth) and place it in the `models/snap_gtr/ckpts/` directory.

Download the **ImageDream Diffusion** checkpoint from here: [ImageDream Checkpoint](https://huggingface.co/Peng-Wang/ImageDream) and place it in the `models/Imagedream/release_models/ImageDream/` directory.

## Usage
To run the spot example:

```bash
cd scripts/
./demo.sh
```

## Acknowledgements
This repositroy is heavily build ontop of **ImageDream Diffusion**, **SnapGTR**, and **DDPM_inversion**. We would like to thank the authors for their open-sourced code and pretrained weights.
