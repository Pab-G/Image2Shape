# ImageDream Diffusion


## TMNT:
```bash
export PYTHONPATH=$PYTHONPATH:./
python3 scripts/demo.py      --image "./assets/tmnt/TMNT_Top_Hat.png"     --text "a teenage mutant ninja turtle with a black hat, 3d asset"     --config_path "./imagedream/configs/sd_v2_base_ipmv.yaml"     --ckpt_path "./release_models/ImageDream/sd-v2.1-base-4view-ipmv.pt"     --mode "pixel"     --num_frames 5 --method Inversion
```

## Spot: 
```bash
export PYTHONPATH=$PYTHONPATH:./
python3 scripts/demo.py      --image "./assets/spot/spot.png"     --text "a cow with a black hat, 3d asset"     --config_path "./imagedream/configs/sd_v2_base_ipmv.yaml"     --ckpt_path "./release_models/ImageDream/sd-v2.1-base-4view-ipmv.pt"     --mode "pixel"     --num_frames 5 --method Inversion
```

## Knight: 
```bash
export PYTHONPATH=$PYTHONPATH:./
python3 scripts/demo.py      --image "./assets/knight/Knight_Cowboy_Hat.png"     --text "a knight with a sword and a brown cowboy hat, 3d asset"     --config_path "./imagedream/configs/sd_v2_base_ipmv.yaml"     --ckpt_path "./release_models/ImageDream/sd-v2.1-base-4view-ipmv.pt"     --mode "pixel"     --num_frames 5 --method Inversion
```
## Giraffe: 
```bash
export PYTHONPATH=$PYTHONPATH:./
python3 scripts/demo.py      --image "./assets/giraffe/front_edit.png"     --text "a giraffe with a pink scarf around its neck, 3d asset"     --config_path "./imagedream/configs/sd_v2_base_ipmv.yaml"     --ckpt_path "./release_models/ImageDream/sd-v2.1-base-4view-ipmv.pt"     --mode "pixel"     --num_frames 5 --method Inversion
```

## Humanoid: 
```bash
export PYTHONPATH=$PYTHONPATH:./
python3 scripts/demo.py      --image "./assets/humanoid/front_edit.png"     --text "a human with a pink scarf around his neck, 3d asset"     --config_path "./imagedream/configs/sd_v2_base_ipmv.yaml"     --ckpt_path "./release_models/ImageDream/sd-v2.1-base-4view-ipmv.pt"     --mode "pixel"     --num_frames 5 --method Inversion
```

```bash
export PYTHONPATH=$PYTHONPATH:./
python3 scripts/demo.py      --image "./assets/humanoid/sun_glasses.png"     --text "a human with a pink sunglasses, 3d asset"     --config_path "./imagedream/configs/sd_v2_base_ipmv.yaml"     --ckpt_path "./release_models/ImageDream/sd-v2.1-base-4view-ipmv.pt"     --mode "pixel"     --num_frames 5 --method Inversion
```