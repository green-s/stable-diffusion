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
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import streamlit as st


config_path = "configs/stable-diffusion/v1-inference.yaml"
ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def gen_random_seed():
    return random.randrange(0, np.iinfo(np.uint32).max)


@st.experimental_singleton
def get_device_name():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_sampler(sampler_name, model, device):
    if sampler_name == "PLMS":
        return PLMSSampler(model, device=device)
    elif sampler_name == "DDIM":
        return DDIMSampler(model, device=device)
    elif sampler_name == "k_dpm_2_a":
        return KSampler(model, "dpm_2_ancestral", device=device)
    elif sampler_name == "k_dpm_2":
        return KSampler(model, "dpm_2", device=device)
    elif sampler_name == "k_euler_a":
        return KSampler(model, "euler_ancestral", device=device)
    elif sampler_name == "k_euler":
        return KSampler(model, "euler", device=device)
    elif sampler_name == "k_heun":
        return KSampler(model, "heun", device=device)
    elif sampler_name == "k_lms":
        return KSampler(model, "lms", device=device)
    else:
        return KSampler(model, "lms", device=device)


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
        device = torch.device(get_device_name())
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        model.to(device)
        model.eval()
        model.half()
        return model
    except AttributeError:
        raise SystemExit


def decode_image(model, img):
    x_samples_ddim = model.decode_first_stage(img)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255.0 * rearrange(x_samples_ddim[0].cpu().numpy(), "c h w -> h w c")
    return x_sample.astype(np.uint8)


def save_image(
    image,
    prompt,
    seed,
    width,
    height,
    steps,
    cfg_scale,
    normalize_prompt_weights,
    sampler_name,
):
    metadata = PngInfo()
    metadata.add_text("SD prompt", str(prompt))
    metadata.add_text("SD seed", str(seed))
    metadata.add_text("SD width", str(width))
    metadata.add_text("SD height", str(height))
    metadata.add_text("SD steps", str(steps))
    metadata.add_text("SD cfg_scale", str(cfg_scale))
    metadata.add_text("SD normalize_prompt_weights", str(normalize_prompt_weights))
    metadata.add_text("SD sampler_name", str(sampler_name))
    image.save(f"outputs/{time.strftime('%Y%m%d-%H%M%S')}_{seed}.png", pnginfo=metadata)


def check_prompt_length(model, prompt):
    tokenizer = model.cond_stage_model.tokenizer
    max_length = model.cond_stage_model.max_length

    info = model.cond_stage_model.tokenizer(
        [prompt],
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt",
    )
    ovf = info["overflowing_tokens"][0]
    overflowing_count = ovf.shape[0]
    if overflowing_count == 0:
        return

    vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    overflowing_words = [vocab.get(int(x), "") for x in ovf]
    overflowing_text = tokenizer.convert_tokens_to_string("".join(overflowing_words))

    st.warning(
        f"Too many input tokens. ({len(overflowing_words)}) have been truncated: {overflowing_text}"
    )


st.set_page_config(
    page_title="Stable Diffusion",
    layout="wide",
)
st.title("Stable Diffusion")
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with st.sidebar:
    random_seed = st.checkbox("Random", False, key="random_seed")
    seed_col1, seed_col2 = st.columns([2, 1])
    with seed_col1:
        seed = st.number_input(
            "Seed",
            0,
            2**32,
            gen_random_seed()
            if random_seed
            else int(st.session_state.seed_value)
            if "seed_value" in st.session_state
            else 42,
            key="seed",
            disabled=random_seed,
        )
        st.session_state.seed_value = seed
    with seed_col2:
        incr_seed = st.empty()
        decr_seed = st.empty()
    iteration_seeds = st.selectbox(
        "Iteration Seeds",
        ["Random", "Subsequent"],
        help="Random uses a pseudorandom (deterministic) seed for each subsequent iteration. Subsequent increases the seed by 1 for each subsequent iteration.",
    )
    iterations = st.number_input("Iterations", 1, value=8, key="iterations")

    def incr_seed_it():
        st.session_state.seed_value += iterations

    def decr_seed_it():
        st.session_state.seed_value -= iterations

    incr_seed.button("-Imgs", on_click=decr_seed_it, disabled=random_seed)
    decr_seed.button("+Imgs", on_click=incr_seed_it, disabled=random_seed)
    sampler_name = st.selectbox(
        "Sampler",
        [
            "DDIM",
            "PLMS",
            "k_lms",
            "k_heun",
            "k_euler",
            "k_euler_a",
            "k_dpm_2",
            "k_dpm_2_a",
        ],
        2,
        key="sampler",
    )
    steps_col1, steps_col2, steps_col3 = st.columns([4, 4, 1])
    steps = steps_col1.number_input(
        "Steps",
        15,
        value=50,
        key="steps",
        help="More steps increases the chance of convergence and reduces noise.",
    )
    use_regenerate_steps = steps_col3.checkbox("", True, key="use_regenerate_steps")
    regenerate_steps = steps_col2.number_input(
        "Regen Steps",
        15,
        value=150,
        key="regenerate_steps",
        disabled=not use_regenerate_steps,
        help="Steps to use during image regeneration.",
    )
    scale = st.number_input(
        "Scale",
        1.0,
        value=7.5,
        key="scale",
        help="Adherence to the prompt. Higher values require more steps and can produce color artifacts.",
    )
    eta_scale = st.number_input(
        "Eta Scale",
        0.0,
        1.0,
        value=0.0,
        key="eta_scale",
        help="Determines the mix between the sampler and DDPM. 0.0 is the sampler, 1.0 is DDPM. Higher values increase noise and require more steps.",
    )
    skip_normalize = st.checkbox(
        "Skip prompt weight normalization", False, key="skip_normalize"
    )
    width_col, height_col = st.columns(2)
    with width_col:
        width = st.number_input("Width", 1, value=512, key="width")
    with height_col:
        height = st.number_input("Height", 1, value=512, key="height")
    cols = st.number_input("Columns", 1, value=4, key="cols")
    render_intermediates = st.checkbox(
        "Render intermediates", True, key="render_intermediates"
    )
    refresh_interval = st.number_input(
        "Refresh interval",
        1,
        int(steps),
        value=15,
        key="refresh_interval",
        disabled=not render_intermediates,
        help="How many steps to wait between refreshing the image. Lower values may decrease performance.",
    )
    init_col, init_offset_col = st.columns(2)
    with init_col:
        render_initial = st.checkbox(
            "Render initial",
            True,
            key="render_initial",
            disabled=not render_intermediates,
            help="Whether to render an initial iteration in addition to those at the interval.",
        )
    with init_offset_col:
        init_offset = st.number_input(
            "Offset",
            0,
            value=1,
            key="init_offset",
            disabled=not render_intermediates,
            help="Offset step of the initial image to render.",
        )

seed = int(seed)
steps = (
    int(steps)
    if not (
        "regenerate" in st.session_state
        and st.session_state.regenerate
        and use_regenerate_steps
    )
    else int(regenerate_steps)
)
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
ddim_eta = float(eta_scale)
latent_channels = 4
downsampling_factor = 8
sampler = get_sampler(sampler_name, model, get_device_name())

prompt_form_slot = st.container()

# TODO support higher batch sizes
base_count = 0
image_count = batch_size * int(iterations)
cols = int(cols)
rows = math.ceil(image_count / cols)
image_cols = st.columns(cols)
image_widget_containers = [image_cols[i % cols].container() for i in range(image_count)]
image_widgets = []
blank_image = Image.new("RGB", (width, height))
for i, w in enumerate(image_widget_containers):
    k = f"image_{i}"
    k_s = f"{k}_seed"
    e = w.empty()
    image_widgets.append(e)
    if k in st.session_state:
        e.image(
            st.session_state[k],
            caption=st.session_state[k_s] if k_s in st.session_state else "",
            output_format="PNG",
        )
    else:
        e.image(blank_image, output_format="PNG")

    def save_callback(k=k, k_s=k_s):
        save_image(
            Image.fromarray(st.session_state[k]),
            st.session_state.prompt if "prompt" in st.session_state else "",
            st.session_state[k_s] if k_s in st.session_state else "",
            width,
            height,
            steps,
            scale,
            not skip_normalize,
            sampler_name,
        )

    w.button(
        "Save",
        on_click=save_callback,
        key=f"{k}_render",
    )

    def regenerate_callback(i=i):
        st.session_state.generate = True
        st.session_state.regenerate = True
        st.session_state.regenerate_i = i

    w.button("Regenerate", on_click=regenerate_callback, key=f"{k}_regenerate")


def update_image(i, widgets, model, image, caption=None):
    image = decode_image(model, image)
    st.session_state[f"image_{i}"] = image
    widgets[i].image(image, output_format="PNG", caption=caption)


with prompt_form_slot:
    with st.form("prompt_form"):
        prompt = st.text_input(
            "Prompt",
            value=st.session_state.prompt_text
            if "prompt_text" in st.session_state
            else "",
            key="prompt",
        )

        def apply_callback():
            st.session_state.regenerate = False
            st.session_state.prompt_text = st.session_state.prompt

        def generate_callback():
            st.session_state.generate = True
            apply_callback()

        st.form_submit_button(
            "Apply",
            on_click=apply_callback,
            help="Save the prompt without generating. Will stop generation in progress.",
        )
        st.form_submit_button("Generate", on_click=generate_callback)
    progress = st.progress(0.0)

    if prompt == "":
        st.error("Please enter a prompt")
        st.stop()
    elif "generate" not in st.session_state or not st.session_state.generate:
        st.stop()
    else:
        st.session_state.prompt_text = prompt
    check_prompt_length(model, prompt)

st.session_state.generate = False

data = [int(batch_size) * [prompt]]

tic = time.time()

with torch.no_grad():
    with precision_scope(get_device_name()), model.ema_scope():
        if "regenerate" in st.session_state and st.session_state.regenerate:
            iters = trange(
                st.session_state.regenerate_i,
                st.session_state.regenerate_i + 1,
                desc="Sampling",
            )
            k_s = f"image_{st.session_state.regenerate_i}_seed"
            if k_s in st.session_state:
                seed = st.session_state[k_s]
            elif iteration_seeds == "Random":
                for n in range(st.session_state.regenerate_i):
                    seed = gen_random_seed()
            else:
                seed = seed + st.session_state.regenerate_i
        else:
            iters = trange(iterations, desc="Sampling")

        for n in iters:
            seed_everything(seed)
            st.session_state[f"image_{n}_seed"] = seed
            for prompts in tqdm(data, desc="data", dynamic_ncols=True):
                progress.progress(
                    float(n) / float(batch_size * iterations - 1)
                    if not (
                        "regenerate" in st.session_state and st.session_state.regenerate
                    )
                    else 1.0
                )
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
                            model.get_learned_conditioning(
                                batch_size * [subprompts[i]]
                            ),
                            alpha=weight,
                        )
                else:  # just standard 1 prompt
                    c = model.get_learned_conditioning(prompts)

                shape = [
                    latent_channels,
                    height // downsampling_factor,
                    width // downsampling_factor,
                ]

                img_callback = None
                if render_intermediates:

                    def img_callback(img, i):
                        if i % refresh_interval == 0 or (
                            render_initial and i == init_offset
                        ):
                            update_image(n, image_widgets, model, img, seed)

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

                update_image(n, image_widgets, model, samples_ddim, seed)

                del samples_ddim

            if iteration_seeds == "Random":
                seed = gen_random_seed()
            else:
                seed += 1

    toc = time.time()
    print(f"{batch_size * iterations} images generated in", "%4.2fs" % (toc - tic))
