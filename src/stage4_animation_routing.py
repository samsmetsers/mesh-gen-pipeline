import os
import subprocess
import shutil
import sys

class AnimationRouter:
    def __init__(self, unirig_path="./UniRig"):
        """
        Routes the refined mesh to either character rigging (UniRig) or rigid object kinematics (Blender).
        """
        self.unirig_path = os.path.abspath(unirig_path)

    def process_character(self, mesh_path, output_path="output/rigged_character.glb"):
        """
        Executes UniRig autoregressive skeleton prediction and skinning weight prediction.
        """
        print(f"Processing character rigging for {mesh_path} using UniRig...")
        
        abs_mesh_path = os.path.abspath(mesh_path)
        abs_output_dir = os.path.abspath(os.path.dirname(output_path))
        python_exe = sys.executable
        
        base_name = os.path.splitext(os.path.basename(mesh_path))[0]
        extraction_dir = os.path.join(abs_output_dir, base_name)
        
        # Step 1: Extraction (using trimesh instead of Blender)
        print("Step 1: Extracting mesh data...")
        extract_script = os.path.abspath("src/extract_trimesh.py")
        subprocess.run([
            python_exe, extract_script,
            "--input", abs_mesh_path,
            "--output_dir", abs_output_dir
        ], check=True)
        
        # Step 2: Skeleton Prediction
        print("Step 2: Skeleton prediction...")
        skel_task = "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
        subprocess.run([
            python_exe, "run.py",
            f"--task={skel_task}",
            f"--input={abs_mesh_path}",
            f"--output_dir={abs_output_dir}",
            "--npz_dir=tmp"
        ], cwd=self.unirig_path, check=True)
        
        # Step 3: Skinning Prediction
        print("Step 3: Skinning weight prediction...")
        skin_task = "configs/task/quick_inference_unirig_skin.yaml"
        subprocess.run([
            python_exe, "run.py",
            f"--task={skin_task}",
            f"--input={abs_mesh_path}",
            f"--output_dir={abs_output_dir}",
            "--npz_dir=tmp"
        ], cwd=self.unirig_path, check=True)
        
        # Step 4: High-Poly Reskinning (Calculation outside Blender)
        print("Step 4: Calculating high-poly skin weights...")
        high_poly_npz = os.path.join(extraction_dir, "raw_data.npz")
        sampled_skin_npz = os.path.join(extraction_dir, "predict_skin.npz")
        high_poly_skin_npz = os.path.join(extraction_dir, "high_poly_skin.npz")
        
        reskin_script = os.path.abspath("src/calculate_high_poly_skin.py")
        subprocess.run([
            python_exe, reskin_script,
            "--high_poly_npz", high_poly_npz,
            "--sampled_skin_npz", sampled_skin_npz,
            "--output_npz", high_poly_skin_npz
        ], check=True)
        
        # Step 5: Final Assembly in Blender
        print("Step 5: Final assembly in Blender...")
        skel_npz = os.path.join(extraction_dir, "predict_skeleton.npz")
        assemble_script = os.path.abspath("src/assemble_character.py")
        
        blender_cmd = [
            "blender", 
            "--background", 
            "--python", assemble_script, 
            "--", 
            "--mesh", abs_mesh_path,
            "--skel_npz", skel_npz,
            "--skin_npz", high_poly_skin_npz,
            "--output", os.path.abspath(output_path),
            "--test_anim"
        ]
        subprocess.run(blender_cmd, check=True)
        
        print(f"Character rigging complete. Output: {output_path}")
        return output_path

    def process_rigid_object(self, mesh_path, output_path="output/animated_scene.glb"):
        """
        Executes Blender headless script for procedural rigid-body kinematics.
        """
        print(f"Processing rigid object animation for {mesh_path} using Blender...")
        
        blender_script = os.path.abspath("src/blender_rigid_body.py")
        blender_cmd = [
            "blender", 
            "--background", 
            "--python", blender_script, 
            "--", 
            "--input", os.path.abspath(mesh_path),
            "--output", os.path.abspath(output_path)
        ]
        
        print(f"Executing: {' '.join(blender_cmd)}")
        subprocess.run(blender_cmd, check=True)
        
        print(f"Rigid object animation saved to {output_path}")
        return output_path

if __name__ == "__main__":
    ar = AnimationRouter()
    # Mocking a decision
    ar.process_rigid_object("output/refined_mesh.obj")
