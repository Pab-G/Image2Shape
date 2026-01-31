import torch
import os
import math
from tqdm import tqdm
import numpy as np


def load_real_image(folder="data/",
                    img_name=None,
                    idx=0,
                    img_size=512,
                    device='cuda'):
    from ddm_inversion.utils import pil_to_tensor
    from PIL import Image
    from glob import glob
    if img_name is not None:
        path = os.path.join(folder, img_name)
    else:
        path = glob(folder + "*")[idx]

    img = Image.open(path).resize((img_size, img_size))

    img = pil_to_tensor(img).to(device)

    if img.shape[1] == 4:
        img = img[:, :3, :, :]
    return img


def mu_tilde(model, xt, x0, timestep):
    "mu_tilde(x_t, x_0) DDPM paper eq. 7"
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[
        prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    alpha_t = model.scheduler.alphas[timestep]
    beta_t = 1 - alpha_t
    alpha_bar = model.scheduler.alphas_cumprod[timestep]
    return ((alpha_prod_t_prev**0.5 * beta_t) /
            (1 - alpha_bar)) * x0 + ((alpha_t**0.5 * (1 - alpha_prod_t_prev)) /
                                     (1 - alpha_bar)) * xt


def sample_xts_from_x0(model,
                       x0,
                       num_inference_steps=50,
                       sampler=None,
                       shape=None):
    """
    Samples from P(x_1:T|x_0)
    """
    #torch.manual_seed(42)
    sqrt_alpha_bar = sampler.sqrt_alphas_cumprod
    sqrt_one_minus_alpha_bar = sampler.sqrt_one_minus_alphas_cumprod
    timesteps = torch.from_numpy(
        sampler.ddim_timesteps[::-1].copy()).long().to(model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros((num_inference_steps + 1, 5, shape[0], shape[1],
                       shape[2])).to(x0.device)
    xts[0] = x0
    for t in reversed(timesteps):
        idx = num_inference_steps - t_to_idx[int(t)]
        # Generate noise for one image and repeat for all in the batch
        #shared_noise = torch.randn(
        #    (1, *x0.shape[1:]), device=x0.device).repeat(x0.shape[0], 1, 1, 1)
        xts[idx] = x0 * sqrt_alpha_bar[t] + torch.randn_like(
            x0) * sqrt_one_minus_alpha_bar[t]

    return xts


def encode_text(model, prompts):
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_encoding = model.text_encoder(
            text_input.input_ids.to(model.device))[0]
    return text_encoding


def forward_step(model, model_output, timestep, sample):
    next_timestep = min(
        model.scheduler.config.num_train_timesteps - 2,
        timestep + model.scheduler.config.num_train_timesteps //
        model.scheduler.num_inference_steps)

    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    # alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep] if next_ltimestep >= 0 else self.scheduler.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t**
                            (0.5) * model_output) / alpha_prod_t**(0.5)

    # 5. TODO: simple noising implementatiom
    next_sample = model.scheduler.add_noise(pred_original_sample, model_output,
                                            torch.LongTensor([next_timestep]))
    return next_sample


def get_variance(model, timestep, sampler):
    # Beta Tilde_t  = (1 - \tilde{alpha_{t-1}}) / (1 - \tilde{alpha_t}) * beta_t

    # get t-1
    prev_timestep = timestep - sampler.ddpm_num_timesteps // len(
        sampler.ddim_timesteps)
    # get alpha for t
    alpha_prod_t = sampler.alphas_cumprod[timestep]
    # get alpha for t-1
    alpha_prod_t_prev = sampler.alphas_cumprod[
        prev_timestep] if prev_timestep >= 0 else sampler.alphas_cumprod[0]

    # calculate 1 - alphas
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    # compute variance schedule for t
    variance = (beta_prod_t_prev /
                beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance


def inversion_forward_process(model,
                              x0,
                              etas=None,
                              prog_bar=False,
                              prompt=None,
                              cfg_scale=3.5,
                              num_inference_steps=50,
                              eps=None,
                              timesteps=None,
                              shape=None,
                              sampler=None,
                              c_=None,
                              uc_=None,
                              seed=42):
    variance_noise_shape = (num_inference_steps, 5, shape[0], shape[1],
                            shape[2])

    if torch.any(sampler.ddim_sigmas):
        eta_is_zero = False
    else:
        eta_is_zero = True
    if type(etas) in [int, float]:
        etas = [etas] * num_inference_steps

    # Noisy latents for each timestep
    xts = sample_xts_from_x0(model,
                             x0,
                             num_inference_steps=num_inference_steps,
                             sampler=sampler,
                             shape=shape)
    # Compute zs
    alpha_bar = sampler.alphas_cumprod
    zs = torch.zeros(size=variance_noise_shape, device=model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0
    op = tqdm(timesteps) if prog_bar else timesteps
    batch = 5

    for t in op:
        idx = num_inference_steps - t_to_idx[int(t)] - 1
        t_tensor = torch.full((batch, ), t, device=xt.device, dtype=torch.long)

        # -----------------------------------------------------------------
        # Define some akin to ImageDream
        sqrt_one_minus_alphas = sampler.ddim_sqrt_one_minus_alphas
        sqrt_one_minus_at = torch.full((batch, 1, 1, 1),
                                       sqrt_one_minus_alphas[idx],
                                       device=xt.device)
        alphas = sampler.ddim_alphas
        alphas_prev = sampler.ddim_alphas_prev
        sigmas = sampler.ddim_sigmas
        a_t = torch.full((batch, 1, 1, 1), alphas[idx], device=xt.device)
        a_prev = torch.full((batch, 1, 1, 1),
                            alphas_prev[idx],
                            device=xt.device)
        sigma_t = torch.full((batch, 1, 1, 1), sigmas[idx], device=xt.device)
        # -----------------------------------------------------------------

        # Predict noise residual
        if not eta_is_zero:
            xt = xts[idx + 1]
        with torch.no_grad():
            out_cond = model.apply_model(x_noisy=xt, t=t_tensor, cond=c_)
            out_uncond = model.apply_model(x_noisy=xt, t=t_tensor, cond=uc_)
            out = out_uncond + cfg_scale * (out_cond - out_uncond)
        noise_pred = out

        # Actual x_{t-1} where we must land
        xtm1 = xts[idx]

        # compute predicted x0 given current xt
        pred_original_sample = (xt -
                                sqrt_one_minus_at * noise_pred) / a_t.sqrt()

        # calculate t-1
        prev_timestep = t - sampler.ddpm_num_timesteps // len(
            sampler.ddim_timesteps)

        # lookup alpha prod x_{t-1}
        alpha_prod_t_prev = sampler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else sampler.alphas_cumprod[0]

        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * noise_pred

        # mean of x_{t-1}
        mu_xt = alpha_prod_t_prev**(0.5) * pred_original_sample + dir_xt

        # get noise trajcetory
        z = (xtm1 - mu_xt) / sigma_t

        zs[idx] = z

        # correction to avoid numerical precision issues
        xtm1 = mu_xt + sigma_t * z
        xts[idx] = xtm1

    # NOTE: Test whether its better with or without using this:
    if not zs is None:
        zs[0] = torch.zeros_like(zs[0])

    # NOTE: The conditioning view should be same at every timestep: but maybe have a look what infulence it has if we noise it too?
    xts[:, 4] = xts[0, 4]
    zs[:, 4, :, :, :] *= 0.0  # no noise-correction for conditioning view
    # If there are some memory issues maybe try this:
    xt_final = xt.detach()
    zs_final = zs.detach()
    xts_final = [x.detach() for x in xts]
    return xt, zs, xts


def reverse_step(model,
                 model_output,
                 timestep,
                 sample,
                 eta=0,
                 variance_noise=None,
                 sampler=None,
                 idx=None):
    batch = 5
    sqrt_one_minus_alphas = sampler.ddim_sqrt_one_minus_alphas
    sqrt_one_minus_at = torch.full((batch, 1, 1, 1),
                                   sqrt_one_minus_alphas[idx],
                                   device=model.device)
    alphas = sampler.ddim_alphas
    alphas_prev = sampler.ddim_alphas_prev
    sigmas = sampler.ddim_sigmas
    a_t = torch.full((batch, 1, 1, 1), alphas[idx], device=model.device)
    a_prev = torch.full((batch, 1, 1, 1),
                        alphas_prev[idx],
                        device=model.device)
    sigma_t = torch.full((batch, 1, 1, 1), sigmas[idx], device=model.device)
    pred_original_sample_2 = (sample -
                              sqrt_one_minus_at * model_output) / a_t**(0.5)

    prev_timestep = timestep - sampler.ddpm_num_timesteps // len(
        sampler.ddim_timesteps)

    dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * model_output

    alpha_prod_t_prev = sampler.alphas_cumprod[
        prev_timestep] if prev_timestep >= 0 else sampler.alphas_cumprod[0]

    mu_xt = alpha_prod_t_prev**(0.5) * pred_original_sample_2 + dir_xt

    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape,
                                         device=model.device)
            print(
                "Warning: sampling random noise for inversion reverse process")
            exit()
        sigma_z = sigma_t * variance_noise
        mu_xt = mu_xt + sigma_z

    return mu_xt


def inversion_reverse_process(model,
                              xT,
                              etas=0,
                              c_=None,
                              uc_=None,
                              cfg_scales=None,
                              prog_bar=False,
                              zs=None,
                              controller=None,
                              asyrp=False,
                              sampler=None):

    batch_size = 5
    if etas is None: etas = 0
    if type(etas) in [int, float]:
        etas = [etas] * len(sampler.ddim_timesteps)
    assert len(etas) == len(sampler.ddim_timesteps)
    # Timesteps that we are sampling 1, 21, 41, ... depending on num_inference_steps
    timesteps = torch.tensor(sampler.ddim_timesteps).to(model.device)
    xt = xT

    ts = torch.flip(timesteps[:zs.shape[0]], dims=[0])
    op = tqdm(ts) if prog_bar else ts

    t_to_idx = {int(v): k for k, v in enumerate(ts)}
    imgs = []
    for t in op:
        # t are the timesteps from last to first e.g. 981, 961, 941, ... depending on num_inference_steps
        idx = len(sampler.ddim_timesteps) - t_to_idx[int(t)] - (
            len(sampler.ddim_timesteps) - zs.shape[0]) - 1
        t_tensor = torch.full((batch_size, ),
                              t,
                              device=xt.device,
                              dtype=torch.long)

        # Unconditional embedding
        with torch.no_grad():
            uncond_out = model.apply_model(x_noisy=xt, t=t_tensor, cond=uc_)

        # Conditional embedding
        with torch.no_grad():
            cond_out = model.apply_model(x_noisy=xt, t=t_tensor, cond=c_)

        # Current noise residual
        z = zs[idx] if not zs is None else None

        # Current noise prediction by model
        noise_pred = uncond_out + cfg_scales[0] * (cond_out - uncond_out)

        # Compute less noisy image and set x_t -> x_t-1
        xt = reverse_step(model,
                          noise_pred,
                          t,
                          xt,
                          eta=etas[idx],
                          variance_noise=z,
                          sampler=sampler,
                          idx=idx)
        if controller is not None:
            xt = controller.step_callback(xt)
        imgs.append(xt)
    return xt, zs, imgs
