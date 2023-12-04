# Prediction interface for Cog ⚙️
from cog import BasePredictor, Input, Path
import os
import math
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting, UNet2DConditionModel, AutoencoderKL
from diffusers import (DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler
)

MODEL_CACHE = "cache"
MODEL_INPAINTING_CACHE = "inpainting_cache"
VAE_CACHE = "vae-cache"

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

class Predictor(BasePredictor):
    def setup(self):
        vae = AutoencoderKL.from_single_file(
            "https://huggingface.co/asiryan/juggernaut/blob/main/TRCVAE.safetensors",
            cache_dir=VAE_CACHE
        )

        self.text2img_pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_CACHE,
            vae=vae,
            safety_checker = None, 
            custom_pipeline="lpw_stable_diffusion"
        ).to("cuda")

        self.img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
            MODEL_CACHE,
            vae=vae,
            safety_checker = None, 
            custom_pipeline="lpw_stable_diffusion"
        ).to("cuda")

        unet = UNet2DConditionModel.from_pretrained(MODEL_INPAINTING_CACHE, subfolder="unet", in_channels=9, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
        self.inpainting_pipe = AutoPipelineForInpainting.from_pretrained(
            MODEL_INPAINTING_CACHE, 
            vae=vae,
            unet=unet,
            safety_checker = None
        ).to("cuda")

    def base(self, x):
        return int(8 * math.floor(int(x)/8))

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="((masterpiece)),((best quality)),((high detial)),((realistic,)) Industrial age city, deep canyons in the middle, architectural streets, bazaars, Bridges, rainy days, steampunk, European architecture",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)",
        ),
        image: Path = Input(
            description="Input image for img2img and inpainting modes",
            default=None
        ),
        mask: Path = Input(
            description="Mask image for inpainting mode",
            default=None
        ),
        width: int = Input(
            description="Width", 
            ge=0, 
            le=1920, 
            default=512
        ),
        height: int = Input(
            description="Height", 
            ge=0, 
            le=1920, 
            default=728
        ),
        strength: float = Input(
            description="Strength/weight", 
            ge=0, 
            le=1, 
            default=1
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps", 
            ge=0, 
            le=100, 
            default=20
        ),
        guidance_scale: float = Input(
            description="Guidance scale", 
            ge=0, 
            le=10, 
            default=7.5
        ),
        scheduler: str = Input(
            description="Scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER_ANCESTRAL",
        ),
        use_karras_sigmas: bool = Input(
            description="Use karras sigmas or not", 
            default=False
        ),
        seed: int = Input(
            description="Leave blank to randomize", 
            default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if (seed == 0) or (seed == None):
            seed = int.from_bytes(os.urandom(2), byteorder='big')
        generator = torch.Generator('cuda').manual_seed(seed)
        
        print("Scheduler:", scheduler)
        print("Using karras sigmas:", use_karras_sigmas)
        print("Using seed:", seed)
        
        if image and mask:
            print("Mode: inpainting")
            init_image = Image.open(image).convert('RGB')
            init_mask = Image.open(mask).convert('RGB')
            
            self.inpainting_pipe.scheduler = SCHEDULERS[scheduler].from_config(
                self.inpainting_pipe.scheduler.config, 
                use_karras_sigmas=use_karras_sigmas)

            output_image = self.inpainting_pipe(
                prompt=prompt,
                image=init_image,
                mask_image=init_mask,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=self.base(width),
                height=self.base(height),
                negative_prompt=negative_prompt,
                generator=generator,
            ).images[0]
        elif image:
            print("Mode: img2img")
            init_image = Image.open(image).convert('RGB')

            self.img2img_pipe.scheduler = SCHEDULERS[scheduler].from_config(
                self.img2img_pipe.scheduler.config, 
                use_karras_sigmas=use_karras_sigmas)

            output_image = self.img2img_pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=self.base(width),
                height=self.base(height),
                negative_prompt=negative_prompt,
                generator=generator,
            ).images[0]
        else:
            print("Mode: text2img")
            self.text2img_pipe.scheduler = SCHEDULERS[scheduler].from_config(
                self.text2img_pipe.scheduler.config, 
                use_karras_sigmas=use_karras_sigmas)

            output_image = self.text2img_pipe(
                prompt=prompt,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=self.base(width),
                height=self.base(height),
                negative_prompt=negative_prompt,
                generator=generator,
            ).images[0]
        
        out_path = Path(f"/tmp/output.png")
        output_image.save(out_path)
        return  out_path
