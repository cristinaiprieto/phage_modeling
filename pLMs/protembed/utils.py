# utils.py
import os
import json
import logging
import torch

def rt_dicts(path, strn_or_phg='strain', seq_report=False, test_mode=False):
    """
    Read fasta files and return a dictionary mapping filenames to 
    their sequence content.
    """
    data = {}
    for fname in os.listdir(path):
        if fname.endswith(".faa"):
            with open(os.path.join(path, fname), encoding='utf-8', errors='ignore') as f:
                seq = "".join([line.strip() for line in f if not line.startswith(">")])
                if test_mode:
                    seq = seq[:1000]  # truncate for test
                data[fname] = {"base_pairs": seq}
    return data

def save_to_dir(path, embeddings, pads, strn_or_phage):
    """
    Save embeddings to a directory.
    """
    os.makedirs(path, exist_ok=True)
    torch.save(embeddings, os.path.join(path, f"{strn_or_phage}_embeddings.pt"))
    torch.save(pads, os.path.join(path, f"{strn_or_phage}_pads.pt"))

def complete_n_select(data_dict, context_len):
    chunked = {}
    pad_info = {}

    for key, val in data_dict.items():
        seq = val["base_pairs"]
        chunks = [seq[i:i+context_len] for i in range(0, len(seq), context_len)]
        chunked[key] = chunks
        pad_info[key] = [len(c) for c in chunks]

    return chunked, pad_info
