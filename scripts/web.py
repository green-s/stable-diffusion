import random
import math
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from torch import autocast
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ksampler import KSampler
from ldm.util import instantiate_from_config
import streamlit as st


config_path = "configs/stable-diffusion/v1-inference.yaml"
ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
device = "cuda"


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def gen_random_seed():
    return random.randrange(0, np.iinfo(np.uint32).max)


def split_weighted_subprompts(text):
    """
    grabs all text up to the first occurrence of ':'
    uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
    if ':' has no value defined, defaults to 1.0
    repeats until no text remaining
    """
    remaining = len(text)
    prompts = []
    weights = []
    while remaining > 0:
        if ":" in text:
            idx = text.index(":")  # first occurrence from start
            # grab up to index as sub-prompt
            prompt = text[:idx]
            remaining -= idx
            # remove from main text
            text = text[idx + 1 :]
            # find value for weight
            if " " in text:
                idx = text.index(" ")  # first occurence
            else:  # no space, read to end
                idx = len(text)
            if idx != 0:
                try:
                    weight = float(text[:idx])
                except:  # couldn't treat as float
                    print(
                        f"Warning: '{text[:idx]}' is not a value, are you missing a space?"
                    )
                    weight = 1.0
            else:  # no value found
                weight = 1.0
            # remove from main text
            remaining -= idx
            text = text[idx + 1 :]
            # append the sub-prompt and its weight
            prompts.append(prompt)
            weights.append(weight)
        else:  # no : found
            if len(text) > 0:  # there is still text though
                # take remainder as weight 1
                prompts.append(text)
                weights.append(1.0)
            remaining = 0
    return prompts, weights


@st.experimental_singleton
def instantiate_model(seed):
    seed_everything(seed)
    try:
        config = OmegaConf.load(config_path)
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        model.cuda()
        model.eval()
        model.half()
        model = model.to(device)
        return model
    except AttributeError:
        raise SystemExit


st.set_page_config(
    page_title="Stable Diffusion",
)
st.title("Stable Diffusion")
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with st.sidebar:
    seed_col1, seed_col2 = st.columns(2)
    with seed_col2:
        random_seed = st.checkbox("Random", False, key="random_seed")
    with seed_col1:
        seed = st.number_input(
            "Seed",
            0,
            2**32,
            42 if not random_seed else gen_random_seed(),
            key="seed",
            disabled=random_seed,
        )
    iterations = st.number_input("Iterations", 1, value=12, key="iterations")
    sampler = st.selectbox("Sampler", ["DDIM", "PLMS", "k_lms"], 2, key="sampler")
    steps = st.number_input(
        "Steps", 1, value=50, key="steps", help="More steps = more accurate"
    )
    scale = st.number_input(
        "Scale", 1.0, value=7.5, key="scale", help="Adherence to the prompt"
    )
    skip_normalize = st.checkbox(
        "Skip prompt weight normalization", False, key="skip_normalize"
    )
    width_col, height_col = st.columns(2)
    with width_col:
        width = st.number_input("Width", 1, value=512, key="width")
    with height_col:
        height = st.number_input("Height", 1, value=512, key="height")
    cols = st.number_input("Columns", 1, value=3, key="cols")
    refresh_interval = st.number_input(
        "Refresh interval", 1, int(steps), value=10, key="refresh_interval"
    )
    init_col, init_offset_col = st.columns(2)
    with init_col:
        render_initial = st.checkbox("Render initial", True, key="render_initial")
    with init_offset_col:
        init_offset = st.number_input(
            "Offset",
            0,
            value=1,
            key="init_offset",
            help="Offset of the initial image to render",
        )

with st.form("prompt_form"):
    prompt = st.text_input("Prompt", key="prompt")
    st.form_submit_button("Generate")
progress = st.progress(0.0)

if prompt == "":
    st.error("Please enter a prompt")
    st.stop()

seed = int(seed)
steps = int(steps)
scale = float(scale)
height = int(height)
width = int(width)
batch_size = 1
iterations = int(iterations)
refresh_interval = int(refresh_interval)
init_offset = int(init_offset)

model = instantiate_model(42)
start_code = None
precision_scope = autocast
ddim_eta = 0.0
latent_channels = 4
downsampling_factor = 8
if sampler == "DDIM":
    sampler = DDIMSampler(model)
elif sampler == "PLMS":
    sampler = PLMSSampler(model)
else:
    sampler = KSampler(model)
data = [int(batch_size) * [prompt]]

# TODO support higher batch sizes
base_count = 0
image_count = batch_size * int(iterations)
image_col_count = int(cols)
image_row_count = math.ceil(image_count / image_col_count)
image_cols = st.columns(image_col_count)
image_widgets = [image_cols[i // image_row_count].empty() for i in range(image_count)]
for i, w in enumerate(image_widgets):
    k = f"image_{i}"
    if k in st.session_state:
        w.image(st.session_state[k])


def get_image(model, img):
    x_samples_ddim = model.decode_first_stage(img)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255.0 * rearrange(x_samples_ddim[0].cpu().numpy(), "c h w -> h w c")
    return x_sample.astype(np.uint8)


tic = time.time()

with torch.no_grad():
    with precision_scope("cuda"):
        with model.ema_scope():
            for n in trange(iterations, desc="Sampling"):
                seed_everything(seed)
                for prompts in tqdm(data, desc="data", dynamic_ncols=True):
                    progress.progress(float(n) / float(batch_size * iterations - 1))
                    uc = None
                    if scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    # weighted sub-prompts
                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        # i dont know if this is correct.. but it works
                        c = torch.zeros_like(uc)
                        # get total weight for normalizing
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(0, len(subprompts)):
                            weight = weights[i]
                            if not skip_normalize:
                                weight = weight / totalWeight
                            c = torch.add(
                                c,
                                model.get_learned_conditioning(subprompts[i]),
                                alpha=weight,
                            )
                    else:  # just standard 1 prompt
                        c = model.get_learned_conditioning(prompts)

                    shape = [
                        latent_channels,
                        height // downsampling_factor,
                        width // downsampling_factor,
                    ]

                    def img_callback(img, i):
                        if i % refresh_interval == 0 or (
                            render_initial and i == init_offset
                        ):
                            image = get_image(model, img)
                            st.session_state[f"image_{n}"] = image
                            image_widgets[n].image(image, output_format="PNG")

                    samples_ddim, _ = sampler.sample(
                        S=steps,
                        conditioning=c,
                        batch_size=batch_size,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        x_T=start_code,
                        img_callback=img_callback,
                    )

                    image_widgets[n].image(
                        get_image(model, samples_ddim), output_format="PNG"
                    )

                    del samples_ddim

                seed = gen_random_seed()

    toc = time.time()
    print(f"{batch_size * iterations} images generated in", "%4.2fs" % (toc - tic))
