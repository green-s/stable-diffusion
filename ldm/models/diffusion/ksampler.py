"""wrapper around part of Katherine Crowson's k-diffusion library, making it call compatible with other Samplers"""
import k_diffusion as K
import torch
import torch.nn as nn
from ldm.dream.devices import choose_torch_device
from ldm.models.diffusion.normalize_latent import normalize_latent

class CFGDenoiser(nn.Module):
    def __init__(self, model, rescale=False, rescaling_coeff=1.7):
        super().__init__()
        self.inner_model = model
        self.rescale = rescale
        self.rescaling_coeff = rescaling_coeff

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        x_out = uncond + (cond - uncond) * cond_scale
        if self.rescale:
            x_out = normalize_latent(x_out, self.rescaling_coeff)
        return x_out


class KSampler(object):
    def __init__(
        self,
        model,
        schedule='lms',
        device=None,
        rescale=False,
        rescaling_coeff=1.7,
        **kwargs
    ):
        super().__init__()
        self.model = K.external.CompVisDenoiser(model)
        self.schedule = schedule
        self.device   = device or choose_torch_device()
        self.rescale = rescale
        self.rescaling_coeff = rescaling_coeff

        def forward(self, x, sigma, uncond, cond, cond_scale):
            x_in = torch.cat([x] * 2)
            sigma_in = torch.cat([sigma] * 2)
            cond_in = torch.cat([uncond, cond])
            uncond, cond = self.inner_model(
                x_in, sigma_in, cond=cond_in
            ).chunk(2)
            return uncond + (cond - uncond) * cond_scale

    # most of these arguments are ignored and are only present for compatibility with
    # other samples
    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):

        sigmas = self.model.get_sigmas(S)
        if x_T is not None:
            x = x_T * sigmas[0]
        else:
            x = (
                torch.randn([batch_size, *shape], device=self.device)
                * sigmas[0]
            )   # for GPU draw
        model_wrap_cfg = CFGDenoiser(self.model, self.rescale, self.rescaling_coeff)
        extra_args = {
            'cond': conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': unconditional_guidance_scale,
        }
        _callback = None
        if callback is not None and img_callback is not None:
            def _callback(kargs):
                callback(kargs['i'])
                img_callback(kargs['x'], kargs['i'])
        elif img_callback is not None:
            def _callback(kargs):
                img_callback(kargs['x'], kargs['i'])
        elif callback is not None:
            def _callback(kargs):
                callback(kargs['i'])
        return (
            K.sampling.__dict__[f'sample_{self.schedule}'](
                model_wrap_cfg, x, sigmas, callback=_callback, extra_args=extra_args
            ),
            None,
        )
