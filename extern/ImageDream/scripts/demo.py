import os
import sys
import argparse
from unittest import skip
from PIL import Image, ImageDraw, ImageFont

import numpy as np
from IPython.display import display
from omegaconf import OmegaConf
import torch

from imagedream.camera_utils import get_camera
from imagedream.ldm.util import (instantiate_from_config, set_seed,
                                 add_random_background)
from imagedream.ldm.models.diffusion.ddim import DDIMSampler
from imagedream.ldm.models.diffusion.ddm_inversion.inversion_utils import inversion_forward_process, inversion_reverse_process
from imagedream.ldm.models.diffusion.ddm_inversion.ptp_classes import AttentionStore, show_cross_attention
from imagedream.ldm.models.diffusion.ddm_inversion.ptp_utils import register_attention_control
from imagedream.model_zoo import build_model
from torchvision import transforms as T

import open_clip
# Some visualization utilities from ChatGPT:


def visualize_noising_trajectory(model, wts, num_views=5, num_steps=50):
    """
    wts: List of tensors [step0, step1, ..., stepT]
    num_views: 5 (Front, Right, Back, Left, Conditioning)
    """
    # 1. Select 5 equidistant indices: x0, ~x12, ~x25, ~x37, xT
    indices = [
        0, num_steps // 4, num_steps // 2, (3 * num_steps) // 4, num_steps
    ]

    view_rows = []

    with torch.no_grad():
        for v in range(num_views):
            step_images = []
            for idx in indices:
                # Extract the specific view's latent at the specific step
                # wts[idx] shape is [batch, channels, h, w] -> [5, 4, 64, 64]
                latent = wts[idx][v:v + 1]

                # 2. Decode latent to image
                # ImageDream/Stable Diffusion VAE scaling factor is usually 0.18215
                img_tensor = model.decode_first_stage(latent)

                # 3. Post-process to PIL
                img = (img_tensor / 2 + 0.5).clamp(0, 1)
                img = img.cpu().permute(0, 2, 3, 1).numpy()
                img = (img[0] * 255).astype(np.uint8)
                step_images.append(Image.fromarray(img))

            # Concatenate the 5 steps horizontally for this view
            view_row = np.hstack([np.asarray(i) for i in step_images])
            view_rows.append(view_row)

    # 4. Stack all view rows vertically
    full_grid = np.vstack(view_rows)
    return Image.fromarray(full_grid)


def save_tensor_views(tensor, save_path="views_debug.png", labels=None):
    """
    tensor: (N, C, H, W) in range [-1, 1]
    labels: List of strings for each view (e.g., ["Left", "Right", "Back"])
    """
    # 1. Denormalize: [-1, 1] -> [0, 1]
    grid_tensor = (tensor.detach().cpu() * 0.5) + 0.5
    grid_tensor = torch.clamp(grid_tensor, 0, 1)

    # 2. Convert to list of PIL images to add text
    pil_images = []
    for i in range(grid_tensor.shape[0]):
        # Convert single tensor to PIL
        img_np = (grid_tensor[i].permute(1, 2, 0).numpy() * 255).astype(
            np.uint8)
        img_pil = Image.fromarray(img_np)

        # Add Label if provided
        if labels and i < len(labels):
            draw = ImageDraw.Draw(img_pil)
            # Use default font (or path to a .ttf if you have one)
            draw.text((10, 10), f"View {i}: {labels[i]}", fill=(255, 255, 0))

        pil_images.append(img_pil)

    # 3. Concatenate horizontally
    widths, heights = zip(*(i.size for i in pil_images))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in pil_images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save(save_path)
    print(f"Visualization saved to: {save_path}")


def load_and_stack_views(paths, device="cuda"):
    """
    paths: List of strings [path_to_right, path_to_back, path_to_left]
    """
    transform = T.Compose([
        T.Resize((args.size, args.size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGBA")
        img = add_random_background(img)
        tensors.append(transform(img))

    return torch.stack(tensors, dim=0).to(device)


def i2i_new(model,
            image_size,
            prompt,
            uc,
            sampler,
            ip=None,
            step=100,
            batch_size=8,
            skip=36,
            cfg_src=3.5,
            cfg_tar=15.0,
            xa=0.6,
            sa=0.2,
            ddim_eta=0.0,
            dtype=torch.float32,
            device="cuda",
            camera=None,
            num_frames=4,
            pixel_control=False,
            transform=None):
    """ The function supports additional image prompt.
    Args:
        model (_type_): the image dream model
        image_size (_type_): size of diffusion output
        prompt (_type_): text prompt for the image
        uc (_type_): _description_
        sampler (_type_): _description_
        ip (Image, optional): the image prompt. Defaults to None.
        step (int, optional): _description_. Defaults to 20.
        scale (float, optional): _description_. Defaults to 7.5.
        batch_size (int, optional): _description_. Defaults to 8.
        ddim_eta (float, optional): _description_. Defaults to 0.0.
        dtype (_type_, optional): _description_. Defaults to torch.float32.
        device (str, optional): _description_. Defaults to "cuda".
        camera (_type_, optional): _description_. Defaults to None.
        num_frames (int, optional): _description_. Defaults to 4
        pixel_control: whether to use pixel conditioning. Defaults to False.
    """
    #----------------------------------------------------------------------
    # Original prompts:

    # spot
    #prompt_src = ['a cow, 3d asset']

    # knight
    #prompt_src = ['a knight with a sword, 3d asset']

    # tmnt
    #prompt_src = ['teenage mutant ninja turtle, 3d asset'] 
    
    #giraffe: 
    #prompt_src = ['a giraffe, 3d asset']
    
    #humanoid:
    prompt_src = ['a human, 3d asset']
    #----------------------------------------------------------------------
    prompt_src = model.get_learned_conditioning(prompt_src).to(device).repeat(
        batch_size, 1, 1)

    if type(prompt) != list:
        prompt = [prompt]

    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        c = model.get_learned_conditioning(prompt).to(device)
        c_ = {"context": c.repeat(batch_size, 1, 1)}
        uc_ = {"context": uc.repeat(batch_size, 1, 1)}

        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames

        ip_embed = None
        if ip is not None:
            ip_embed = model.get_learned_image_conditioning(ip).to(device)
            ip_ = ip_embed.repeat(batch_size, 1, 1)
            c_["ip"] = ip_
            uc_["ip"] = torch.zeros_like(ip_)
        if pixel_control:
            assert camera is not None
            ip = transform(ip).to(device)
            input_tensor = ip[None, :, :, :].to(dtype=torch.float32)
            ip_img = model.get_first_stage_encoding(
                model.encode_first_stage(input_tensor))
            c_["ip_img"] = ip_img
            uc_["ip_img"] = torch.zeros_like(ip_img)

        shape = [4, image_size // 8, image_size // 8]
        #----------------------------------------------------------------------
        # Load assets for other views:

        # spot:
        #torch_array = load_and_stack_views([
        #    "./assets/spot/spot_left.png", "./assets/spot/spot_right.png",
        #    "./assets/spot/spot_back.png"
        #],
        #                                   device=device)

        # knight:
        #torch_array = load_and_stack_views([
        #    "./assets/knight/knight_right_view.png", "./assets/knight/knight_left_view.png",
        #    "./assets/knight/knight_back_view.png"
        #],
        #                                   device=device)

        #tmnt:
        #torch_array = load_and_stack_views([
        #    "./assets/tmnt/tmnt_right_view.png",
        #    "./assets/tmnt/tmnt_left_view.png",
        #    "./assets/tmnt/tmnt_back_view.png"
        #],
        #                                   device=device)
        
        # giraffe:
        #torch_array = load_and_stack_views([
        #    "./assets/giraffe/right.png",
        #    "./assets/giraffe/left.png",
        #    "./assets/giraffe/back.png"
        #],
        #                                   device=device)
        
        #humanoid: 
        torch_array = load_and_stack_views([
            "./assets/humanoid/right.png",
            "./assets/humanoid/left.png",
            "./assets/humanoid/back.png"      
        ],
                                             device=device)
        
        #----------------------------------------------------------------------
        # Visualisation:
        #save_tensor_views(torch_array,
        #                  save_path="./diffusion_out/debug/check_input_views.png",
        #                  labels=["Left", "Right", "Back"])

        encode_array = model.get_first_stage_encoding(
            (model.encode_first_stage(torch_array.to(device))))

        # Tricky because views might need to be arranged for now try out:
        #TMNT:
        #x0 = torch.cat((encode_array[0].unsqueeze(0), ip_img,
        #                encode_array[1].unsqueeze(0),
        #                encode_array[2].unsqueeze(0), ip_img),
        #   
        # dim=0)
        #x0 = torch.cat((encode_array[1].unsqueeze(0), ip_img,
        #                encode_array[0].unsqueeze(0),
        #                encode_array[2].unsqueeze(0), ip_img),
        #               dim=0)
        #Giraffe:
        x0 = torch.cat((encode_array[1].unsqueeze(0), ip_img,
                        encode_array[0].unsqueeze(0),
                        encode_array[2].unsqueeze(0), ip_img),
                       dim=0)
        
        # DDPM-inversion: Forward
        eta = 0.5  #TMNT: 1.0
        sampler.make_schedule(step, ddim_eta=eta)
        with torch.no_grad():
            _, zs, wts = inversion_forward_process(
                model,
                x0=x0,
                etas=eta,
                prompt=prompt_src,
                cfg_scale=cfg_src,
                prog_bar=True,
                num_inference_steps=step,
                timesteps=sampler.ddim_timesteps[::-1],
                shape=shape,
                sampler=sampler,
                c_=c_,
                uc_=uc_,
            )

        # Visualisation:
        #visualize_noising_trajectory(
        #    model, wts, num_views=5, num_steps=step).save(
        #        "./diffusion_out/debug/debug_noising_trajectory.png")

        wt = wts[-1]

        with torch.no_grad():
            #controller = AttentionStore()
            #register_attention_control(model, controller)
            xt, _, imgs = inversion_reverse_process(model,
                                                 xT=wts[step - skip],
                                                 etas=eta,
                                                 c_=c_,
                                                 uc_=uc_,
                                                 cfg_scales=[cfg_tar],
                                                 zs=zs[:(step - skip)],
                                                 sampler=sampler,
                                                 controller=None)
            #x = visualize_noising_trajectory(model, imgs, num_views=5, num_steps=step - skip - 1)
            #x.save(f"output.png")
        x_sample = model.decode_first_stage(xt)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255.0 * x_sample.permute(0, 2, 3, 1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


def i2i(model,
        image_size,
        prompt,
        uc,
        sampler,
        ip=None,
        step=20,
        scale=5.0,
        batch_size=8,
        ddim_eta=0.0,
        dtype=torch.float32,
        device="cuda",
        camera=None,
        num_frames=4,
        pixel_control=False,
        transform=None):
    """ The function supports additional image prompt.
    Args:
        model (_type_): the image dream model
        image_size (_type_): size of diffusion output
        prompt (_type_): text prompt for the image
        uc (_type_): _description_
        sampler (_type_): _description_
        ip (Image, optional): the image prompt. Defaults to None.
        step (int, optional): _description_. Defaults to 20.
        scale (float, optional): _description_. Defaults to 7.5.
        batch_size (int, optional): _description_. Defaults to 8.
        ddim_eta (float, optional): _description_. Defaults to 0.0.
        dtype (_type_, optional): _description_. Defaults to torch.float32.
        device (str, optional): _description_. Defaults to "cuda".
        camera (_type_, optional): _description_. Defaults to None.
        num_frames (int, optional): _description_. Defaults to 4
        pixel_control: whether to use pixel conditioning. Defaults to False.
    """
    if type(prompt) != list:
        prompt = [prompt]

    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        c = model.get_learned_conditioning(prompt).to(device)
        c_ = {"context": c.repeat(batch_size, 1, 1)}
        uc_ = {"context": uc.repeat(batch_size, 1, 1)}

        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames

        if ip is not None:
            ip_embed = model.get_learned_image_conditioning(ip).to(device)
            ip_ = ip_embed.repeat(batch_size, 1, 1)
            c_["ip"] = ip_
            uc_["ip"] = torch.zeros_like(ip_)

        if pixel_control:
            assert camera is not None
            ip = transform(ip).to(device)
            ip_img = model.get_first_stage_encoding(
                model.encode_first_stage(ip[None, :, :, :]))
            c_["ip_img"] = ip_img
            uc_["ip_img"] = torch.zeros_like(ip_img)

        shape = [4, image_size // 8, image_size // 8]
        samples_ddim, _ = sampler.sample(
            S=step,
            conditioning=c_,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_,
            eta=0.0,
            x_T=None,
        )
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255.0 * x_sample.permute(0, 2, 3, 1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


class ImageDreamDiffusion():

    def __init__(self, args) -> None:
        assert args.mode in ["pixel", "local"]
        assert args.num_frames % 4 == 1 if args.mode == "pixel" else True

        set_seed(args.seed)
        dtype = torch.float16 if args.fp16 else torch.float32
        device = args.device
        batch_size = max(4, args.num_frames)

        print("load image dream diffusion model ... ")
        model = build_model(args.model_name,
                            config_path=args.config_path,
                            ckpt_path=args.ckpt_path)
        model.device = device
        model.to(device)
        model.eval()

        neg_texts = "uniform low no texture ugly, boring, bad anatomy, blurry, pixelated,  obscure, unnatural colors, poor lighting, dull, and unclear."
        sampler = DDIMSampler(model)
        uc = model.get_learned_conditioning([neg_texts]).to(device)
        print("image dream model load done . ")

        # pre-compute camera matrices
        if args.use_camera:
            camera = get_camera(num_frames=4,
                                elevation=5,
                                azimuth_start=0,
                                azimuth_span=360,
                                extra_view=(args.mode == "pixel"))
            camera = camera.repeat(batch_size // args.num_frames, 1).to(device)
        else:
            camera = None

        self.image_transform = T.Compose([
            T.Resize((args.size, args.size)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.dtype = dtype
        self.device = device
        self.batch_size = batch_size
        self.args = args
        self.model = model
        self.sampler = sampler
        self.uc = uc
        self.camera = camera

    def diffuse(self, t, ip, n_test=3):
        images = []
        if self.args.method == "Inversion":
            print("USING INVERSION METHOD")
            for _ in range(n_test):
                img = i2i_new(self.model,
                              self.args.size,
                              t,
                              self.uc,
                              self.sampler,
                              ip=ip,
                              step=100, #spot: 100, #TMNT: 100
                              skip=18, #spot: 13, #TMNT: 36
                              cfg_src=1.0, #spot: 1.0, # TMNT: 1.0
                              cfg_tar=2.0, #spot: 3.5, #TMNT: 1.9
                              xa=0.6,
                              sa=0.2,
                              batch_size=self.batch_size,
                              ddim_eta=0.0,
                              dtype=self.dtype,
                              device=self.device,
                              camera=self.camera,
                              num_frames=args.num_frames,
                              pixel_control=(args.mode == "pixel"),
                              transform=self.image_transform)
                img = np.concatenate(img[:4], 1)
                images.append(img)
        elif self.args.method == "ImageDream":
            print("INFO: USING IMAGEDREAM METHOD")
            for _ in range(n_test):
                img = i2i(self.model,
                          self.args.size,
                          t,
                          self.uc,
                          self.sampler,
                          ip=ip,
                          step=100,
                          scale=5.0,
                          batch_size=self.batch_size,
                          ddim_eta=0.0,
                          dtype=self.dtype,
                          device=self.device,
                          camera=self.camera,
                          num_frames=args.num_frames,
                          pixel_control=(args.mode == "pixel"),
                          transform=self.image_transform)
                img = np.concatenate(img[:4], 1)
                images.append(img)
        else:
            raise NotImplementedError(
                f"method <<{self.args.method}>> not implemented! \n please choose from [ImageDream, Inversion]"
            )
        return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="sd-v2.1-base-4view-ipmv",
        help="load pre-trained model from hugginface",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="load model from local config (override model_name)",
    )
    parser.add_argument("--ckpt_path",
                        type=str,
                        default=None,
                        help="path to local checkpoint")
    parser.add_argument("--text",
                        type=str,
                        default="an astronaut riding a horse")
    parser.add_argument("--image", type=str, default="./assets/astrounaut.png")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_frames",
                        type=int,
                        default=5,
                        help=" \
        num of frames (views) to generate, should be in [4 or 5],  \
        5 for pixel control, 4 for local control")
    parser.add_argument("--use_camera", type=int, default=1)
    parser.add_argument("--camera_elev", type=int, default=5)
    parser.add_argument("--camera_azim", type=int, default=90)
    parser.add_argument("--camera_azim_span", type=int, default=360)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode",
                        type=str,
                        default="pixel",
                        help="ip mode default pixel")
    parser.add_argument("--method",
                        type=str,
                        default="ImageDream",
                        help="which method to use: ImageDream or Inversion")
    args = parser.parse_args()

    t = args.text + args.suffix
    assert args.num_frames in [4, 5], "num_frames should be in [4, 5]"
    assert os.path.exists(args.image), "image does not exist!"
    ip = Image.open(args.image).convert("RGBA")
    ip = add_random_background(ip)

    image_dream = ImageDreamDiffusion(args)
    image_dream.model.to(torch.float32)
    #image_dream.model.first_stage_model.to(torch.float32)

    images = image_dream.diffuse(t, ip, n_test=1)

    name = os.path.basename(args.image).split(".")[0]
    images = np.concatenate(images, 0)
    #Image.fromarray(images).save(
    #    f"diffusion_out/{args.method}/{name}_{args.mode}_dream.png")
    #print(f"saved image under diffusion_out as: {name}_{args.mode}_dream.png")
    Image.fromarray(images).save(f"{args.method}_{name}.png")
    print(f"saved image: {args.method}_{name}.png")
