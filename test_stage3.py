import os
import pymeshlab
from src.stage3_mesh_refinement import MeshRefiner

def create_mock_mesh(path):
    """Creates a simple cube OBJ for testing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n")
        f.write("v 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n")
        f.write("f 1 2 3 4\nf 8 7 6 5\nf 1 5 6 2\nf 2 6 7 3\nf 3 7 8 4\nf 4 8 5 1\n")

def test_refinement():
    raw_path = "output/test_raw.obj"
    refined_path = "output/test_refined.obj"
    
    print("Creating mock mesh...")
    create_mock_mesh(raw_path)
    
    print("Initializing MeshRefiner...")
    mr = MeshRefiner()
    
    try:
        print("Starting refinement...")
        # Target a very low face count for this small mock mesh
        mr.refine(raw_path, refined_path, target_face_count=10)
        
        if os.path.exists(refined_path):
            print(f"SUCCESS: Refined mesh saved to {refined_path}")
            # Check face count
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(refined_path)
            print(f"Final face count: {ms.current_mesh().face_number()}")
        else:
            print("FAILURE: Refined mesh file not found.")
            
    except Exception as e:
        print(f"ERROR during refinement: {e}")

if __name__ == "__main__":
    test_refinement()
