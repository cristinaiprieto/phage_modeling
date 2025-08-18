#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Generate embeddings using ESM2
"""

import argparse
import pathlib
import numpy as np
import os
import glob
import pandas as pd

import torch

from esm import Alphabet, ProteinBertModel, pretrained, MSATransformer

class FastaBatchedDataset(object):
    def __init__(self, sequence_labels, sequence_strs):
        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = [s.rstrip('*') for s in sequence_strs]  # Ensure no sequence ends with '*'

    @classmethod
    def from_file(cls, fasta_file):
        sequence_labels, sequence_strs = [], []
        cur_seq_label = None
        buf = []

        def _flush_current_seq():
            nonlocal cur_seq_label, buf
            if cur_seq_label is None:
                return
            sequence = "".join(buf).rstrip('*')  # Remove '*' from the end if present
            sequence_labels.append(cur_seq_label)
            sequence_strs.append(sequence)
            cur_seq_label = None
            buf = []

        with open(fasta_file, "r") as infile:
            for line_idx, line in enumerate(infile):
                if line.startswith(">"):  # label line
                    _flush_current_seq()
                    line = line[1:].strip()
                    if len(line) > 0:
                        cur_seq_label = line
                    else:
                        cur_seq_label = f"seqnum{line_idx:09d}"
                else:  # sequence line
                    buf.append(line.strip())

        _flush_current_seq()

        assert len(set(sequence_labels)) == len(sequence_labels), "Found duplicate sequence labels"

        return cls(sequence_labels, sequence_strs)

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, idx):
        return self.sequence_labels[idx], self.sequence_strs[idx]

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.sequence_strs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()
        return batches


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract protein-level embeddings for sequences in FAA files based on IDs from a CSV file"
    )

    parser.add_argument(
        "model_location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "input_dir",
        type=pathlib.Path,
        help="Directory containing FAA files to process",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="Output directory for extracted embeddings",
    )
    parser.add_argument(
        "csv_file",
        type=pathlib.Path,
        help="CSV file containing IDs of strains to process",
    )
    parser.add_argument(
        "id_column",
        type=str,
        help="Column name in the CSV file containing strain IDs",
    )

    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1022,
        help="truncate sequences longer than the given value",
    )
    parser.add_argument("--nogpu", 
                        action="store_true", 
                        help="Do not use GPU even if available")
    parser.add_argument("--num_threads", 
                        type=int, 
                        default=1, 
                        help="Number of threads for PyTorch operations")
    return parser


def process_fasta_file(model, alphabet, fasta_file, output_file, args):
    """Process a single fasta file and save the protein embeddings."""
    dataset = FastaBatchedDataset.from_file(fasta_file)
    if len(dataset) == 0:
        print(f"No sequences found in {fasta_file}, skipping...")
        return
    
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=batches
    )
    print(f"Processing {fasta_file} with {len(dataset)} sequences")

    # Only generate protein-level embeddings (mean of residue embeddings)
    protein_embs = {}
    selected_layer = model.num_layers - 1  # Use the last layer by default

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"  Batch {batch_idx + 1} of {len(batches)} ({toks.size(0)} sequences)")
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=[selected_layer])

            for i, label in enumerate(labels):
                truncate_len = min(args.truncation_seq_length, len(strs[i]))
                # Extract embeddings (skip the first token which is BOS)
                layer_embeddings = out["representations"][selected_layer].cpu().numpy()[i, 1:truncate_len + 1]
                # Calculate mean across the token dimension
                mean_embeddings = np.mean(layer_embeddings, axis=0)
                protein_embs[label] = mean_embeddings

    # Save only protein embeddings
    np.save(output_file, protein_embs)
    print(f"Saved protein embeddings to {output_file}")


def run(args):
    torch.set_num_threads(args.num_threads)
    
    # Load model
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError("This script currently does not handle models with MSA input (MSA Transformer).")
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load strain IDs from CSV file
    try:
        df = pd.read_csv(args.csv_file)
        if args.id_column not in df.columns:
            raise ValueError(f"Column '{args.id_column}' not found in CSV file")
        
        csv_strain_ids = set(df[args.id_column].unique())
        print(f"Found {len(csv_strain_ids)} unique strain IDs in CSV")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Find all available .faa files in the input directory
    input_files = glob.glob(os.path.join(args.input_dir, "*.faa"))
    available_strains = {os.path.splitext(os.path.basename(f))[0] for f in input_files}
    
    # Find strains that exist in both CSV and input directory
    strains_to_process = csv_strain_ids.intersection(available_strains)
    missing_strains = csv_strain_ids - available_strains
    
    if missing_strains:
        print(f"Warning: {len(missing_strains)} strains from CSV not found in input directory")
        if len(missing_strains) <= 10:
            print(f"Missing strains: {', '.join(missing_strains)}")
        else:
            print(f"First 10 missing strains: {', '.join(list(missing_strains)[:10])}")
    
    # Check which strains are already processed
    already_processed = []
    to_process = []
    
    for strain_id in strains_to_process:
        output_file = args.output_dir / f"{strain_id}.npy"
        if os.path.exists(output_file):
            already_processed.append(strain_id)
        else:
            to_process.append(strain_id)
    
    # Print summary before processing
    print("\nSummary:")
    print(f"Total strains in CSV: {len(csv_strain_ids)}")
    print(f"Strains found in input directory: {len(strains_to_process)}")
    print(f"Strains already processed: {len(already_processed)}")
    print(f"Strains to process: {len(to_process)}")
    
    if not to_process:
        print("All available strains have already been processed. Nothing to do.")
        return
    
    print(f"\nStarting processing of {len(to_process)} strains...\n")
    
    # Process each strain
    for idx, strain_id in enumerate(to_process):
        faa_file = os.path.join(args.input_dir, f"{strain_id}.faa")
        output_file = args.output_dir / f"{strain_id}.npy"
        
        print(f"File {idx+1}/{len(to_process)} ({len(to_process)-idx-1} remaining)")
        process_fasta_file(model, alphabet, faa_file, output_file, args)
    
    print(f"\nProcessing complete. {len(to_process)} strains processed.")


def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()