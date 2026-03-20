import torch
import os
import sys
import cv2
from PIL import Image

# Add TRELLIS.2 to the python path so we can import trellis2 and o_voxel
TRELLIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "TRELLIS.2"))
if TRELLIS_DIR not in sys.path:
    sys.path.append(TRELLIS_DIR)

# These must be set before certain imports if using OpenCV with EXR
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

    def generate_mesh(self, image_path, output_mesh_path="output/raw_mesh.obj", target_face_count=1000000):
        """
        Processes the image into a 3D asset.
        """
        if not HAS_TRELLIS:
            print(f"Generating mock 3D mesh from {image_path}...")
            with open(output_mesh_path, "w") as f:
                f.write("# Raw Mesh Placeholder from Stage 2")
            return output_mesh_path

        print(f"Generating 3D mesh from {image_path} using TRELLIS.2...")
        image = Image.open(image_path)
        
        # Run pipeline
        outputs = self.pipeline.run(image)
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
            texture_size        =   2048,
            remesh              =   True,
            remesh_band         =   1,
            remesh_project      =   0,
            verbose             =   True
        )
        
        if output_mesh_path.endswith(".obj"):
            glb_mesh.export(output_mesh_path)
        else:
            glb_mesh.export(output_mesh_path)

        print(f"Mesh saved to {output_mesh_path}")
        return output_mesh_path

if __name__ == "__main__":
    tg = TrellisGenerator()
    ref_img = "output/reference.png"
    if not os.path.exists(ref_img):
        print(f"Creating mock reference image at {ref_img}")
        os.makedirs("output", exist_ok=True)
        Image.new('RGB', (512, 512), color = (73, 109, 137)).save(ref_img)
    
    tg.generate_mesh(ref_img)
