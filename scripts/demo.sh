#!/bin/bash

# 1. Generate the 4 views with ImageDream
# Using the spot example
export PYTHONPATH=$PYTHONPATH:../models/ImageDream/
conda run -n imagedream python ../models/ImageDream/scripts/demo.py --image "../demo/assets/tmnt/front_edit.png"     --text "a teenage mutant ninja turtle with a black hat, 3d asset"     --config_path "../models/ImageDream/imagedream/configs/sd_v2_base_ipmv.yaml"     --ckpt_path "../models/ImageDream/release_models/ImageDream/sd-v2.1-base-4view-ipmv.pt"     --mode "pixel"     --num_frames 5 --method Inversion

# 2. Reconstruct 3D with SnapGTR
conda run -n gtr python3 ../models/snap_gtr/scripts/prepare_mv.py --in_dir ../demo/outputs/ImageDreamDiffusion/Inversion_front_edit.png --out_dir ../models/snap_gtr/examples/tmnt
conda run -n gtr python3 ../models/snap_gtr/scripts/inference.py --ckpt_path ../models/snap_gtr/ckpts/full_checkpoint.pth --in_dir ../models/snap_gtr/examples/tmnt/ --out_dir ../demo/outputs/3DAssets/tmnt
