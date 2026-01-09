import os
import sys
import argparse
from PIL import Image
import numpy as np
from IPython.display import display
from omegaconf import OmegaConf
import torch

from imagedream.camera_utils import get_camera
from imagedream.ldm.util import (instantiate_from_config, set_seed,
                                 add_random_background)
from imagedream.ldm.models.diffusion.ddim import DDIMSampler
from imagedream.ldm.models.diffusion.ddm_inversion.inversion_utils import inversion_forward_process, inversion_reverse_process
from imagedream.model_zoo import build_model
from torchvision import transforms as T


def latent_to_image(model, latent, steps=10, sampler=None):

    imgs = []
    step_size = max(1, latent.shape[0] // steps)
    with torch.inference_mode():
        indices = list(range(0, latent.shape[0], step_size))
        if indices[-1] != latent.shape[0] - 1:
            indices.append(latent.shape[0] - 1)

        for i in indices:
            z = latent[i:i + 1]
            z = latent[i, 3].unsqueeze(0)
            x_rec = model.decode_first_stage(z)
            x_rec = torch.clamp((x_rec + 1.0) / 2.0, min=0.0, max=1.0)
            x_rec = 255.0 * x_rec.permute(0, 2, 3, 1).cpu().numpy()
            imgs.append(x_rec.astype(np.uint8)[0])

    from PIL import Image
    img_strip = np.concatenate(imgs, axis=1)
    #Image.fromarray(img_strip).show()
    Image.fromarray(img_strip).save(
        "/home/yulong/pvbg-thesis/ImageDream/extern/ImageDream/Ivisualisation.png"
    )
    print("saved image")


def load_sub_images(image_path, num_splits=4):
    # 1. Load the full concatenated image
    # PIL loads as (Width, Height)
    full_image = Image.open(image_path)

    # 2. Convert to PyTorch Tensor
    # transforms.ToTensor() converts PIL image to (C, H, W) and scales to [0.0, 1.0]
    to_tensor = T.ToTensor()
    tensor_image = to_tensor(full_image)

    # 3. Split the tensor along the width dimension
    # Format is (C, H, W), so Width is dimension 2
    # This reverses: np.concatenate(img[:4], 1)
    sub_images = torch.chunk(tensor_image, chunks=num_splits, dim=2)

    # 4. Stack them into a single batch tensor
    # Final shape will be: (4, C, H, W)
    batch_array = torch.stack(sub_images)

    return batch_array


def load_and_stack_views(left_path, right_path, back_path):
    """
    Loads three images, applies the transform pipeline to each,
    and stacks them into (3, C, H, W).
    """
    image_transform = T.Compose([
        T.Resize((args.size, args.size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # 1. Load images using PIL
    # Ensuring RGB mode is critical for the Normalize step to work on 3 channels
    left_img = Image.open(left_path).convert('RGB')
    right_img = Image.open(right_path).convert('RGB')
    back_img = Image.open(back_path).convert('RGB')

    # 2. Apply your existing self.image_transform to each PIL image
    # This Resize -> ToTensor -> Normalize sequence now gets the PIL input it wants
    left_tensor = image_transform(left_img)
    right_tensor = image_transform(right_img)
    back_tensor = image_transform(back_img)

    # 3. Stack along a new 0th dimension
    # Resulting shape: (3, C, H, W)
    stacked_views = torch.stack([left_tensor, right_tensor, back_tensor],
                                dim=0)

    return stacked_views


def load_and_stack_ndarrays(left_path, right_path, back_path):
    """
    Loads three image paths, converts them to ndarrays, 
    and stacks them into shape (3, C, H, W).
    """
    paths = [left_path, right_path, back_path]
    images = []

    for path in paths:
        # Load image and ensure it's RGB (3 channels)
        img = Image.open(path).convert('RGB')

        # Convert to ndarray (H, W, C)
        img_array = np.array(img)

        # Change layout from (H, W, C) to (C, H, W)
        # transpose(2, 0, 1) moves the channel axis to the front
        img_array = img_array.transpose(2, 0, 1)

        images.append(img_array)

    # Stack into a single array: (3, C, H, W)
    stacked_views = np.stack(images, axis=0)

    return stacked_views


def i2i_new(model,
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

        torch_array = load_sub_images(
            "/home/yulong/pvbg-thesis/ImageDream/extern/ImageDream/astronaut_pixel_dream_old.png",
            num_splits=4)

        #torch_array = load_and_stack_views("./assets/spot_left.png",
        #                                   "./assets/spot_right.png",
        #                                   "./assets/spot_back.png")

        #torch_array = transform(torch_array).to(device)
        torch_array = torch_array * 2.0 - 1.0  # to [-1, 1]

        #img = (torch_array[3].clamp(0,1)*255).byte()
        #img = img.permute(1,2,0).cpu().numpy()
        #Image.fromarray(img).show()
        #exit()
        #encode_array = model.get_first_stage_encoding(
        #   (model.encode_first_stage(torch_array.to(device))))
        
        #x0 = torch.cat((encode_array[0].unsqueeze(0), ip_img,
        #                encode_array[1].unsqueeze(0),
        #                encode_array[2].unsqueeze(0), ip_img),
        #               dim=0)
        
        #x0 = torch.cat((ip_img, encode_array[0].unsqueeze(0),
        #                encode_array[1].unsqueeze(0),
        #                encode_array[2].unsqueeze(0), ip_img),
        #               dim=0)
        
        x0 = torch.cat((ip_img,
                        model.get_first_stage_encoding(
                            (model.encode_first_stage(
                                torch_array[1:].to(device)))), ip_img),
                       dim=0)

        #ddpm forward:
        eta = 0.9
        sampler.make_schedule(50, ddim_eta=eta)
        wt, zs, wts = inversion_forward_process(
            model,
            x0=x0,
            etas=eta,
            prompt=ip_embed,
            cfg_scale=eta,
            prog_bar=True,
            num_inference_steps=step,
            timesteps=sampler.ddim_timesteps[::-1],
            shape=shape,
            sampler=sampler,
            c_=c_,
            uc_=uc_,
        )
        #from IPython import embed; embed(); exit()
        #latent_to_image(model, wts, steps=5, sampler=sampler)

        # replace wt[4] with ip_img
        wt = wts[-1]
        #wt = torch.cat((wts[-1, :4, :, :, :], ip_img), dim=0)
        zs[:, 4, :, :, :] *= 0.0
        #torch_array = model.decode_first_stage(wts[-1])
        #img = (torch_array[1].clamp(0, 1) * 255).byte()
        #img = img.permute(1, 2, 0).cpu().numpy()
        #Image.fromarray(img).show()

        xt, _, img = inversion_reverse_process(model,
                                               xT=wt,
                                               etas=eta,
                                               c_=c_,
                                               uc_=uc_,
                                               cfg_scales=[scale],
                                               zs=zs,
                                               sampler=sampler)

        #xt, _ = sampler.sample_ddpm(
        #    S=step,
        #    conditioning=c_,
        #    batch_size=batch_size,
        #    shape=shape,
        #    verbose=False,
        #    unconditional_guidance_scale=scale,
        #    unconditional_conditioning=uc_,
        #    eta=eta,
        #    x_T=wt,
        #    zs=zs,
        #)
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
        for _ in range(n_test):
            img = i2i_new(self.model,
                          self.args.size,
                          t,
                          self.uc,
                          self.sampler,
                          ip=ip,
                          step=50,
                          scale=5,
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
    args = parser.parse_args()

    t = args.text + args.suffix
    assert args.num_frames in [4, 5], "num_frames should be in [4, 5]"
    assert os.path.exists(args.image), "image does not exist!"
    ip = Image.open(args.image)
    ip = add_random_background(ip)

    image_dream = ImageDreamDiffusion(args)
    image_dream.model.to(torch.float32)
    #image_dream.model.first_stage_model.to(torch.float32)

    images = image_dream.diffuse(t, ip, n_test=3)

    name = os.path.basename(args.image).split(".")[0]
    images = np.concatenate(images, 0)
    Image.fromarray(images).save(f"{name}_{args.mode}_dream.png")
    print(f"saved image: {name}_{args.mode}_dream.png")
