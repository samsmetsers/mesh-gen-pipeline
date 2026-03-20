import os
import argparse
import re
from src.stage1_vision_prior import VisionPrior
from src.stage2_trellis_wrapper import TrellisGenerator
from src.stage3_mesh_refinement import MeshRefiner
from src.stage4_animation_routing import AnimationRouter

def main():
    parser = argparse.ArgumentParser(description="Mesh Generation Pipeline")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for output files")
    parser.add_argument("--output_name", type=str, help="Base name for output files (e.g. 'owl')")
    parser.add_argument("--type", type=str, choices=["character", "rigid"], default="character", 
                        help="Type of object for animation routing")
    parser.add_argument("--refine", action="store_true", default=True, help="Perform mesh refinement (format conversion)")
    parser.add_argument("--skip_gen", action="store_true", help="Skip image generation and use existing reference.png")
    parser.add_argument("--skip_trellis", action="store_true", help="Skip 3D generation and use existing raw_mesh.obj")
    parser.add_argument("--skip_refine", action="store_true", help="Skip mesh refinement and use existing refined_mesh.glb")
    
    args = parser.parse_args()
    
    # Automatic name generation if not provided
    if args.output_name:
        base_name = args.output_name
    else:
        # Sanitize prompt to create a valid filename
        base_name = re.sub(r'[^\w\s-]', '', args.prompt).strip().lower()
        base_name = re.sub(r'[-\s]+', '_', base_name)[:30]
    
    # Create project-specific subfolder
    project_dir = os.path.join(args.output_dir, base_name)
    os.makedirs(project_dir, exist_ok=True)
    
    # Paths (within project-specific subfolder)
    ref_image_path = os.path.join(project_dir, f"{base_name}_reference.png")
    raw_mesh_path = os.path.join(project_dir, f"{base_name}_raw.obj")
    refined_mesh_path = os.path.join(project_dir, f"{base_name}_refined.glb")
    final_output_path = os.path.join(project_dir, f"{base_name}_final.glb")

    # Stage 1: Vision Prior
    if not args.skip_gen:
        print("\n--- Stage 1: Vision Prior ---")
        vp = VisionPrior()
        vp.generate_reference(args.prompt, ref_image_path)
    else:
        print("\n--- Skipping Stage 1: Using existing reference ---")
        if not os.path.exists(ref_image_path):
            print(f"Error: {ref_image_path} not found.")
            return

    # Stage 2: 3D Generation
    if not args.skip_trellis:
        print("\n--- Stage 2: 3D Generation (TRELLIS.2) ---")
        tg = TrellisGenerator()
        tg.generate_mesh(ref_image_path, raw_mesh_path)
    else:
        print("\n--- Skipping Stage 2: Using existing raw mesh ---")
        if not os.path.exists(raw_mesh_path):
            print(f"Error: {raw_mesh_path} not found.")
            return

    # Stage 3: Mesh Refinement (Format Conversion)
    if not args.skip_refine:
        if args.refine:
            print("\n--- Stage 3: Format Conversion (OBJ to GLB) ---")
            mr = MeshRefiner()
            current_mesh = mr.refine(raw_mesh_path, refined_mesh_path)
        else:
            current_mesh = raw_mesh_path
    else:
        print("\n--- Skipping Stage 3: Using existing refined mesh ---")
        current_mesh = refined_mesh_path
        if not os.path.exists(current_mesh):
            print(f"Error: {current_mesh} not found.")
            return

    # Stage 4: Animation Routing
    print("\n--- Stage 4: Animation Routing ---")
    ar = AnimationRouter()
    if args.type == "character":
        ar.process_character(current_mesh, final_output_path)
    else:
        ar.process_rigid_object(current_mesh, final_output_path)

    print(f"\nPipeline complete! Project files in: {project_dir}")

if __name__ == "__main__":
    main()
