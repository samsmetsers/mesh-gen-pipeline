import torch
from diffusers import DiffusionPipeline, FluxPipeline, Flux2KleinPipeline
from PIL import Image
import os
import argparse

class VisionPrior:
    def __init__(self, model_id="black-forest-labs/FLUX.2-klein-9B"):
        """
        Interfaces with FLUX via diffusers to generate reference images.
        """
        print(f"Loading {model_id}...")

        # Robust loader: try specialized pipelines first if in ID, else fallback
        try:
            if "klein" in model_id:
                self.pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
            else:
                self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        except Exception as e:
            print(f"Specialized loader failed ({e}), falling back to DiffusionPipeline...")
            self.pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

        # Aggressive memory management:
        # Sequential CPU offload is slower but fits almost anything on 10GB
        self.pipe.enable_sequential_cpu_offload()

        # VAE tiling handles the final 1024x1024 decode in smaller chunks
        if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling()

    def generate_reference(self, prompt, output_path="output/reference.png"):
        """
        Generates a clean, orthographic reference image optimized for 3D reconstruction.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Enhanced prompt engineering for 3D priors (condensed for CLIP 77-token limit)
        refined_prompt = (
            f"Full body character sheet of {prompt}, isolated on flat white background. "
            "Orthographic front view, symmetrical T-pose. "
            "Single solid entity, no floating parts, no loose objects, no shadows, "
            "professional 3D asset reference, high detail."
        )

        print(f"Generating image for prompt: {refined_prompt}")

        # Ensure VRAM is as clean as possible
        torch.cuda.empty_cache()

        # Most Flux models (Schnell, Klein) work well with 4 steps
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
def generate_reference_image(prompt, output_path):
    vp = VisionPrior()
    return vp.generate_reference(prompt, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    generate_reference_image(args.prompt, args.output)
