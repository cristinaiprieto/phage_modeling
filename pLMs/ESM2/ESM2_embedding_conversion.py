#!/usr/bin/env python3

import numpy as np
import os
import argparse
from pathlib import Path

def convert_embeddings(input_dir: Path, output_dir: Path):
    """
    Loads dictionary-style ESM-2 embeddings from .npy files, stacks them
    into a single array, and saves them to a new .npy file.

    Args:
        input_dir (Path): Directory containing the source .npy files.
        output_dir (Path): Directory to save the converted .npy files.
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_dir}\n")

    # Process each .npy file in the input directory
    file_list = list(input_dir.glob('*.npy'))
    if not file_list:
        print(f"No .npy files found in {input_dir}. Exiting.")
        return

    for file_path in file_list:
        try:
            print(f"Processing {file_path.name}...")

            # Load the dictionary format embeddings
            embeddings_dict = np.load(file_path, allow_pickle=True).item()

            if not isinstance(embeddings_dict, dict) or not embeddings_dict:
                print(f"  → Skipping {file_path.name}: not a valid dictionary or is empty.")
                continue

            # Extract embeddings and stack them into a single array
            embedding_list = list(embeddings_dict.values())
            embeddings_array = np.stack(embedding_list)

            # Save as a new .npy file
            output_path = output_dir / file_path.name
            np.save(output_path, embeddings_array)

            print(f"  → Converted {file_path.name}: {len(embedding_list)} embeddings, shape {embeddings_array.shape}")

        except Exception as e:
            print(f"  → Error processing {file_path.name}: {str(e)}")

    print(f"\nConversion completed.")

def main():
    """Main function to parse arguments and run the conversion."""
    parser = argparse.ArgumentParser(
        description="Convert ESM-2 dictionary embeddings to stacked NumPy arrays."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing dictionary-style .npy embedding files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory to save the converted stacked-array .npy files.",
    )
    args = parser.parse_args()

    convert_embeddings(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()