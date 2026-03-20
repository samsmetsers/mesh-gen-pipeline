import numpy as np
import os
import argparse
from scipy.spatial import cKDTree

def normalize_vertices(vertices):
    """
    Normalize vertices to [-1, 1] cube, matching UniRig's AugmentAffine.
    """
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    center = (v_max + v_min) / 2
    # Span is max(v_max - v_min) / (1 - (-1)) = max(v_max - v_min) / 2
    span = np.max(v_max - v_min) / 2
    if span < 1e-6: span = 1.0
    
    norm_v = (vertices - center) / span
    return norm_v, center, span

def reskin(sampled_vertices, vertices, sampled_skin, alpha=2.0, threshold=0.03):
    print(f"Reskinning {len(vertices)} vertices using {len(sampled_vertices)} samples...")
    
    # UniRig sampled_vertices are already normalized.
    # We MUST normalize our high-poly vertices to the same space.
    norm_v, _, _ = normalize_vertices(vertices)
    
    # Also, we might need to check if UniRig's normalization used a different reference
    # but usually it's just the mesh itself. 
    # Let's check if the ranges match.
    print(f"Sampled vertices range: {sampled_vertices.min(axis=0)} to {sampled_vertices.max(axis=0)}")
    print(f"Normalized high-poly range: {norm_v.min(axis=0)} to {norm_v.max(axis=0)}")

    tree = cKDTree(sampled_vertices)
    # Using k=7 nearest neighbors
    dis, nearest = tree.query(norm_v, k=7, p=2)
    
    # Weighted sum based on distance
    weights = np.exp(-alpha * dis)
    weight_sum = weights.sum(axis=1, keepdims=True)
    
    # (N, 7, J)
    sampled_skin_nearest = sampled_skin[nearest]
    # (N, J)
    skin = (sampled_skin_nearest * weights[..., np.newaxis]).sum(axis=1) / weight_sum
    
    # Post-processing: threshold and normalize
    mask = (skin >= threshold).any(axis=-1, keepdims=True)
    skin[(skin < threshold) & mask] = 0.0
    sums = skin.sum(axis=-1, keepdims=True)
    sums[sums == 0] = 1.0
    skin = skin / sums
    
    return skin

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--high_poly_npz", required=True)
    parser.add_argument("--sampled_skin_npz", required=True)
    parser.add_argument("--output_npz", required=True)
    args = parser.parse_args()
    
    # Load data
    high_poly_data = np.load(args.high_poly_npz, allow_pickle=True)
    sampled_skin_data = np.load(args.sampled_skin_npz, allow_pickle=True)
    
    # The refined mesh vertices (high poly)
    vertices = high_poly_data['vertices']
    
    # The sampled skinning data (normalized space)
    sampled_vertices = sampled_skin_data['vertices']
    sampled_skin = sampled_skin_data['skin']
    
    # Calculate skin weights for high poly mesh
    high_poly_skin = reskin(
        sampled_vertices=sampled_vertices,
        vertices=vertices,
        sampled_skin=sampled_skin
    )
    
    # Save to NPZ
    np.savez(args.output_npz, skin=high_poly_skin)
    print(f"High poly skin saved to {args.output_npz}")

if __name__ == "__main__":
    main()
