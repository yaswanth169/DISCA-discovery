#!/usr/bin/env python
"""Preprocess tomograms and extract subtomograms."""

import argparse
from pathlib import Path
import numpy as np
import mrcfile
from tqdm import tqdm
from scipy import ndimage


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Cryo-ET tomograms")
    parser.add_argument("--input", type=str, required=True, help="Input tomogram (.mrc)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--box-size", type=int, default=32, help="Subtomogram box size")
    parser.add_argument("--stride", type=int, default=None, help="Sliding window stride")
    parser.add_argument("--particle-pick", action="store_true", help="Use particle picking")
    parser.add_argument("--threshold", type=float, default=1.5, help="Picking threshold (std)")
    parser.add_argument("--min-distance", type=int, default=16, help="Min distance between particles")
    parser.add_argument("--max-particles", type=int, default=10000, help="Max particles to extract")
    parser.add_argument("--normalize", action="store_true", default=True, help="Normalize tomogram")
    return parser.parse_args()


def load_tomogram(path: str) -> np.ndarray:
    print(f"Loading tomogram: {path}")
    with mrcfile.open(path, permissive=True) as mrc:
        if mrc.data is None:
            raise ValueError(f"Could not load data from {path}")
        data = mrc.data.copy()
    
    print(f"   Shape: {data.shape}")
    print(f"   Data type: {data.dtype}")
    print(f"   Value range: [{data.min():.2f}, {data.max():.2f}]")
    return data


def normalize_tomogram(tomo: np.ndarray) -> np.ndarray:
    mean = tomo.mean()
    std = tomo.std()
    if std > 1e-8:
        return (tomo - mean) / std
    return tomo - mean


def particle_picking(tomo: np.ndarray, threshold: float, min_distance: int) -> np.ndarray:
    print(f"Particle picking (threshold: {threshold} std, min_distance: {min_distance})")
    
    high_signal = tomo > (tomo.mean() + threshold * tomo.std())
    coords = np.array(np.where(high_signal)).T
    
    if len(coords) == 0:
        return np.array([])
    
    # Non-maximum suppression
    filtered = []
    coords_list = coords.tolist()
    
    pbar = tqdm(total=len(coords_list), desc="NMS")
    while coords_list:
        current = coords_list.pop(0)
        filtered.append(current)
        
        coords_list = [
            c for c in coords_list
            if np.sqrt(sum((a - b) ** 2 for a, b in zip(c, current))) >= min_distance
        ]
        pbar.update(1)
    pbar.close()
    
    return np.array(filtered)


def sliding_window_positions(shape: tuple, box_size: int, stride: int) -> np.ndarray:
    positions = []
    half = box_size // 2
    
    for z in range(half, shape[0] - half, stride):
        for y in range(half, shape[1] - half, stride):
            for x in range(half, shape[2] - half, stride):
                positions.append([z, y, x])
    
    return np.array(positions)


def extract_subtomograms(tomo: np.ndarray, positions: np.ndarray, box_size: int) -> list:
    print(f"Extracting subtomograms at particle positions")
    print(f"   Box size: {box_size}^3")
    
    half = box_size // 2
    subtomograms = []
    
    for pos in tqdm(positions, desc="Extracting"):
        z, y, x = pos
        
        if (z - half >= 0 and z + half <= tomo.shape[0] and
            y - half >= 0 and y + half <= tomo.shape[1] and
            x - half >= 0 and x + half <= tomo.shape[2]):
            
            subtomo = tomo[z-half:z+half, y-half:y+half, x-half:x+half]
            if subtomo.shape == (box_size, box_size, box_size):
                subtomograms.append(subtomo)
    
    return subtomograms


def save_subtomograms(subtomograms: list, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving subtomograms to: {output_dir}")
    
    for i, subtomo in enumerate(tqdm(subtomograms, desc="Saving")):
        filepath = output_dir / f"subtomo_{i:05d}.mrc"
        with mrcfile.new(str(filepath), overwrite=True) as mrc:
            mrc.set_data(subtomo.astype(np.float32))
    
    print(f"   Saved {len(subtomograms)} files")


def main():
    args = parse_args()
    
    print("=" * 80)
    print("Cryo-ET Tomogram Preprocessing")
    print("=" * 80)
    
    tomo = load_tomogram(args.input)
    
    if args.normalize:
        tomo = normalize_tomogram(tomo)
    
    if args.particle_pick:
        positions = particle_picking(tomo, args.threshold, args.min_distance)
        print(f"   Found {len(positions)} particle locations")
        
        if len(positions) > args.max_particles:
            indices = np.random.choice(len(positions), args.max_particles, replace=False)
            positions = positions[indices]
            print(f"   Sampled {len(positions)} positions")
    else:
        stride = args.stride or args.box_size // 2
        positions = sliding_window_positions(tomo.shape, args.box_size, stride)
        print(f"   Generated {len(positions)} sliding window positions")
    
    subtomograms = extract_subtomograms(tomo, positions, args.box_size)
    print(f"   Extracted {len(subtomograms)} subtomograms")
    
    save_subtomograms(subtomograms, args.output)
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)
    print(f"   Input:  {args.input}")
    print(f"   Output: {args.output}")
    print(f"   Subtomograms: {len(subtomograms)}")


if __name__ == "__main__":
    main()
