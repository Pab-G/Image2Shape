# ImageDream Diffusion

Here are some example commands, skip, eta and cfg have to be tuned per example:
You can also try out running the defualt ImageDream Difffusion by changing method to ImageDream instead of Inversion.


## TMNT:
```bash
export PYTHONPATH=$PYTHONPATH:./
python3 scripts/demo.py      --image "../../../assets/tmnt/front_edit.png"     --text "a teenage mutant ninja turtle with a black hat, 3d asset"     --config_path "./imagedream/configs/sd_v2_base_ipmv.yaml"     --ckpt_path "./release_models/ImageDream/sd-v2.1-base-4view-ipmv.pt"     --mode "pixel"     --num_frames 5 --method Inversion
```

## Spot: 
```bash
export PYTHONPATH=$PYTHONPATH:./
python3 scripts/demo.py      --image "../../../assets/spot/front_edit.png"     --text "a cow with a black hat, 3d asset"     --config_path "./imagedream/configs/sd_v2_base_ipmv.yaml"     --ckpt_path "./release_models/ImageDream/sd-v2.1-base-4view-ipmv.pt"     --mode "pixel"     --num_frames 5 --method Inversion
```

## Knight: 
```bash
export PYTHONPATH=$PYTHONPATH:./
python3 scripts/demo.py      --image "../../../assets/knight/front_edit.png"     --text "a knight with a sword and a brown cowboy hat, 3d asset"     --config_path "./imagedream/configs/sd_v2_base_ipmv.yaml"     --ckpt_path "./release_models/ImageDream/sd-v2.1-base-4view-ipmv.pt"     --mode "pixel"     --num_frames 5 --method Inversion
```
