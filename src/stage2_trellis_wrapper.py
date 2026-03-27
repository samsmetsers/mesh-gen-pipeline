import torch
import os
import sys
import cv2
from PIL import Image
import argparse

# Add TRELLIS.2 to the python path so we can import trellis2 and o_voxel
TRELLIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "TRELLIS.2"))
if TRELLIS_DIR not in sys.path:
    sys.path.append(TRELLIS_DIR)

# These must be set before certain imports if using OpenCV with EXR
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Use sdpa as default attention backend (no flash_attn required)
os.environ.setdefault('ATTN_BACKEND', 'sdpa')

try:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.renderers import EnvMap
    import o_voxel
    HAS_TRELLIS = True
except ImportError:
    print("Warning: TRELLIS.2 dependencies not found. Using placeholder mode.")
    HAS_TRELLIS = False

class TrellisGenerator:
    def __init__(self, model_id="microsoft/TRELLIS.2-4B"):
        """
        Wrapper for TRELLIS.2 3D generation.
        """
        if not HAS_TRELLIS:
            print(f"Loading weights for {model_id} (MOCK)...")
            return

        print(f"Loading TRELLIS.2 pipeline: {model_id}...")
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
        self.pipeline.cuda()
        
        # Load default envmap for PBR
        hdri_path = os.path.join(TRELLIS_DIR, "assets", "hdri", "forest.exr")
        if os.path.exists(hdri_path):
            self.envmap = EnvMap(torch.tensor(
                cv2.cvtColor(cv2.imread(hdri_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                dtype=torch.float32, device='cuda'
            ))
        else:
            print(f"Warning: HDRI not found at {hdri_path}. PBR rendering might fail.")
            self.envmap = None

    def generate_mesh(self, image_path, output_mesh_path="output/raw_mesh.obj", target_face_count=300000):
        """
        Processes the image into a 3D asset.
        Optimized for speed with 8 steps and 512 res.
        """
        if not HAS_TRELLIS:
            print(f"Generating mock 3D mesh from {image_path}...")
            with open(output_mesh_path, "w") as f:
                f.write("# Raw Mesh Placeholder from Stage 2")
            return output_mesh_path

        print(f"Generating 3D mesh from {image_path} using TRELLIS.2 (Speed Optimized)...")
        image = Image.open(image_path)
        
        # Run pipeline with reduced steps for speed
        outputs = self.pipeline.run(
            image,
            sparse_structure_sampler_params={'steps': 8},
            shape_slat_sampler_params={'steps': 8},
            tex_slat_sampler_params={'steps': 8},
            pipeline_type='512'
        )
        mesh = outputs[0]
        
        print("Post-processing with o-voxel...")
        glb_mesh = o_voxel.postprocess.to_glb(
            vertices            =   mesh.vertices,
            faces               =   mesh.faces,
            attr_volume         =   mesh.attrs,
            coords              =   mesh.coords,
            attr_layout         =   mesh.layout,
            voxel_size          =   mesh.voxel_size,
            aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target   =   target_face_count,
            texture_size        =   1024, # Reduced from 2048 for faster baking
            remesh              =   True,
            remesh_band         =   1,
            remesh_project      =   0,
            verbose             =   True
        )
        
        # Always save the GLB (preserves PBR textures/vertex colors)
        glb_path = output_mesh_path.rsplit(".", 1)[0] + ".glb"
        glb_mesh.export(glb_path)
        print(f"GLB saved to {glb_path}")

        # Also save OBJ if requested
        if output_mesh_path.endswith(".obj"):
            glb_mesh.export(output_mesh_path)
            # Try to extract and save the texture image alongside the OBJ
            try:
                from trimesh.visual import TextureVisuals
                vis = glb_mesh.visual
                if isinstance(vis, TextureVisuals) and hasattr(vis, 'material'):
                    mat_img = getattr(vis.material, 'image', None)
                    if mat_img is not None:
                        tex_path = output_mesh_path.rsplit(".", 1)[0] + "_texture.png"
                        mat_img.save(tex_path)
                        print(f"Texture image saved: {tex_path}")
                        # Update MTL file to reference this texture
                        mtl_path = output_mesh_path.rsplit(".", 1)[0] + ".mtl"
                        if os.path.exists(mtl_path):
                            with open(mtl_path, 'a') as f:
                                f.write(f"\nmap_Kd {os.path.basename(tex_path)}\n")
                        print(f"MTL updated with texture reference")
            except Exception as e:
                print(f"Note: could not save texture alongside OBJ: {e}")

        print(f"Mesh saved to {output_mesh_path}")
        return output_mesh_path

class TrellisWrapper(TrellisGenerator):
    """Alias with a generate() interface matching main.py usage."""
    def generate(self, image_path, output_mesh_path):
        return self.generate_mesh(image_path, output_mesh_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    tg = TrellisGenerator()
    tg.generate_mesh(args.image, args.output)
