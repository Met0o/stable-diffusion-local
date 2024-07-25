import torch
import gradio as gr
from huggingface_hub import login
from diffusers import StableDiffusion3Pipeline

login(token="hf_token_here", add_to_git_credential=True)

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

def generate_image(prompt, negative_prompt, num_inference_steps, guidance_scale):
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images[0]
    return image

interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Textbox(label="Negative Prompt", placeholder="Optional"),
        gr.Slider(step=1, minimum=1, maximum=100, value=28, label="Number of Inference Steps"),
        gr.Slider(minimum=1.0, maximum=20.0, step=0.1, value=7.0, label="Guidance Scale")
    ],
    outputs="image",
    title="Stable Diffusion 3 Image Generator",
    description="Generate images with Stable Diffusion 3. Type a prompt and see the magic!"
)

interface.launch(server_name="0.0.0.0", server_port=8912, inbrowser=True, share=False)