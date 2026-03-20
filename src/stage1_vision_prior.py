import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
import os

class VisionPrior:
    def __init__(self, model_id="black-forest-labs/FLUX.2-klein-4B"):
        """
        Interfaces with FLUX.2 via diffusers to generate reference images.
        Uses Flux2KleinPipeline for the 4B 'klein' variant.
        Aggressively optimized for 10GB VRAM on RTX 3080.
        """
        print(f"Loading {model_id}...")
        
        # Load the model in bfloat16 to save space
        self.pipe = Flux2KleinPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        
        # Aggressive memory management:
        # Sequential CPU offload is slower but fits almost anything on 10GB
        self.pipe.enable_sequential_cpu_offload()
        
        # VAE tiling handles the final 1024x1024 decode in smaller chunks
        # Calling directly on the VAE component as the pipeline lacks the wrapper
        if hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling()
        if hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()

    def generate_reference(self, prompt, output_path="output/reference.png"):
        """
        Generates an orthographic, pure white background reference image.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        refined_prompt = (
            f"orthographic view of {prompt}, pure white background, "
            "professional 3D asset reference, high detail, studio lighting"
        )
        
        print(f"Generating image for prompt: {refined_prompt}")
        
        # Ensure VRAM is as clean as possible
        torch.cuda.empty_cache()
        
        image = self.pipe(
            prompt=refined_prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            height=1024,
            width=1024
        ).images[0]
        
        image.save(output_path)
        print(f"Reference image saved to {output_path}")
        return output_path

if __name__ == "__main__":
    vp = VisionPrior()
    vp.generate_reference("a futuristic sci-fi robot character")
