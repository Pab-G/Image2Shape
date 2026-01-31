"""
This code was originally taken from
https://github.com/google/prompt-to-prompt
"""

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
# from IPython.display import display
from tqdm import tqdm


def text_under_image(image: np.ndarray,
                     text: str,
                     text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8)
              for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones(
        (h * num_rows + offset * (num_rows - 1), w * num_cols + offset *
         (num_cols - 1), 3),
        dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset):i * (h + offset) + h:, j * (w + offset):j *
                   (w + offset) + w] = images[i * num_cols + j]

    pil_img = Image.fromarray(image_)
    # display(pil_img)
    return pil_img


def diffusion_step(model,
                   controller,
                   latents,
                   context,
                   t,
                   guidance_scale,
                   low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(
            latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(
            latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input,
                                t,
                                encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    cfg_scales_tensor = torch.Tensor(guidance_scale).view(-1, 1, 1,
                                                          1).to(model.device)
    noise_pred = noise_pred_uncond + cfg_scales_tensor * (
        noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8,
                            width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)

    uncond_input = model.tokenizer([""] * batch_size,
                                   padding="max_length",
                                   max_length=77,
                                   return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(prompt,
                                 padding="max_length",
                                 max_length=77,
                                 return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator,
                                  batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])

    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t,
                                 guidance_scale)

    image = latent2image(model.vqvae, latents)

    return image, latent


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    restored_wt=None,
    restored_zs=None,
    low_resource: bool = False,
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(
        model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer([""] * batch_size,
                                   padding="max_length",
                                   max_length=max_length,
                                   return_tensors="pt")
    uncond_embeddings = model.text_encoder(
        uncond_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator,
                                  batch_size)

    # set timesteps
    # extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps)  #, **extra_set_kwargs)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t,
                                 guidance_scale, low_resource)

    # image = latent2image(model.vae, latents)

    return latents, latent


# THIS FUNCTION WAS MODIFIED FOR ImageDream COMPATIBILITY
#------------------------------------------------------------------------
def register_attention_control(model, controller):

    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        # ImageDream uses 'n_heads' or 'heads' depending on version
        h = getattr(self, "heads", getattr(self, "n_heads", None))

        # We calculate scale once during registration if possible,
        # otherwise we'll handle it inside forward
        scale_attr = getattr(self, "scale", None)

        def forward(x, context=None, mask=None, ip_context=None):
            batch_size, sequence_length, dim = x.shape
            is_cross = context is not None

            # Determine scaling factor
            nonlocal scale_attr
            if scale_attr is None:
                scale_attr = (dim // h)**-0.5

            q = self.to_q(x)
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)

            # Manual Reshaping for MultiViewUNet compatibility
            # [B, L, D] -> [B*H, L, D/H]
            head_dim = dim // h
            q = q.view(batch_size, -1, h,
                       head_dim).transpose(1,
                                           2).reshape(batch_size * h, -1,
                                                      head_dim)
            k = k.view(batch_size, -1, h,
                       head_dim).transpose(1,
                                           2).reshape(batch_size * h, -1,
                                                      head_dim)
            v = v.view(batch_size, -1, h,
                       head_dim).transpose(1,
                                           2).reshape(batch_size * h, -1,
                                                      head_dim)

            # Standard Attention Calculation
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * scale_attr
            attn = sim.softmax(dim=-1)

            # PTP Injection point: Modify attention maps using the controller
            attn = controller(attn, is_cross, place_in_unet)

            out = torch.einsum("b i j, b j d -> b i d", attn, v)

            # Reshape back to [B, L, D]
            out = out.view(batch_size, h, -1,
                           head_dim).transpose(1,
                                               2).reshape(batch_size, -1, dim)

            # --- ImageDream IP-Adapter Support ---
            # Handles the 5th view (reference image) conditioning pass
            if is_cross and ip_context is not None and hasattr(
                    self, 'to_k_ip'):
                ip_k = self.to_k_ip(ip_context)
                ip_v = self.to_v_ip(ip_context)

                ip_k = ip_k.view(batch_size, -1, h,
                                 head_dim).transpose(1, 2).reshape(
                                     batch_size * h, -1, head_dim)
                ip_v = ip_v.view(batch_size, -1, h,
                                 head_dim).transpose(1, 2).reshape(
                                     batch_size * h, -1, head_dim)

                ip_sim = torch.einsum("b i d, b j d -> b i j", q,
                                      ip_k) * scale_attr
                ip_attn = ip_sim.softmax(dim=-1)

                ip_out = torch.einsum("b i j, b j d -> b i d", ip_attn, ip_v)
                ip_out = ip_out.view(batch_size, h, -1, head_dim).transpose(
                    1, 2).reshape(batch_size, -1, dim)

                # Combine standard text-prompt attention with image-prompt attention
                out = out + ip_out

            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, attn, is_cross, place_in_unet):
            return attn

        def __init__(self):
            self.num_att_layers = 0

        def step_callback(self, x):
            return x

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        # Look for the characteristic projections of an attention layer
        if hasattr(net_, 'to_q') and hasattr(net_, 'to_k'):
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1

        if hasattr(net_, 'children'):
            for name, child in net_.named_children():
                count = register_recr(child, count, place_in_unet)
        return count

    cross_att_count = 0

    # ImageDream LatentDiffusionInterface nests the UNet inside model.model.diffusion_model
    target_unet = model.model.diffusion_model

    # Traverse the specific block groups of the MultiViewUNetModel
    if hasattr(target_unet, 'input_blocks'):
        cross_att_count = register_recr(target_unet.input_blocks,
                                        cross_att_count, "down")
    if hasattr(target_unet, 'middle_block'):
        cross_att_count = register_recr(target_unet.middle_block,
                                        cross_att_count, "mid")
    if hasattr(target_unet, 'output_blocks'):
        cross_att_count = register_recr(target_unet.output_blocks,
                                        cross_att_count, "up")

    controller.num_att_layers = cross_att_count
    print(
        f"Successfully registered {cross_att_count} attention layers for ImageDream."
    )


#------------------------------------------------------------------------


def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [
            i for i, word in enumerate(split_text) if word_place == word
        ]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [
            tokenizer.decode([item]).strip("#")
            for item in tokenizer.encode(text)
        ][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha,
                           bounds: Union[float, Tuple[float, float]],
                           prompt_ind: int,
                           word_inds: Optional[torch.Tensor] = None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] *
                                                      alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start:end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(
        prompts,
        num_steps,
        cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
        tokenizer,
        max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1,
                                   len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(
            alpha_time_words, cross_replace_steps["default_"], i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [
                get_word_inds(prompts[i], key, tokenizer)
                for i in range(1, len(prompts))
            ]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(
                        alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1,
                                                len(prompts) - 1, 1, 1,
                                                max_num_words)
    return alpha_time_words
