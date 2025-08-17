# utils.py
import os
import json
import logging
import torch

from Bio import SeqIO
import esm

# utils.py
import os
import logging
import torch
from typing import Dict, Tuple
from Bio import SeqIO

logger = logging.getLogger(__name__)

ALLOWED_AA = set(list("ACDEFGHIKLMNPQRSTVWYBXZOU"))  # keep common extras too (B,Z,X,O,U)

def _clean_protein(seq: str, test_mode: bool = False) -> str:
    """
    Clean a protein sequence for ESM/ProtT5:
      - uppercase
      - remove '*' anywhere (stop codons)
      - remove whitespace
      - replace any non A-Z char with 'X'
      - optionally truncate in test mode
    """
    s = seq.upper().replace("*", "")
    s = "".join(ch for ch in s if ch.isalpha())  # keep A-Z only
    # Map unknowns to X (ESM/ProtT5 both accept X)
    s = "".join(ch if ch in ALLOWED_AA else "X" for ch in s)
    if test_mode:
        s = s[:1000]
    return s

def rt_dicts(path: str, strn_or_phg: str = 'strain', seq_report: bool = False, test_mode: bool = False) -> Dict[str, Dict[str, str]]:
    """
    Read all FASTA/FAA in a directory via Biopython like ESM2 does and
    return {filename_without_ext: {"base_pairs": cleaned_sequence}}.
    Supports .faa, .fa, .fasta (case-insensitive).
    """
    data = {}
    exts = {".faa", ".fa", ".fasta"}
    files = [f for f in os.listdir(path) if os.path.splitext(f)[1].lower() in exts]
    if not files:
        logger.warning(f"No FASTA files with extensions {exts} found in {path}")

    for fname in files:
        fpath = os.path.join(path, fname)
        try:
            # Concatenate all records in a file into one long sequence (your prior behavior).
            # If you instead want per-record outputs, adapt workflow to handle multiple per file.
            rec_seqs = []
            for rec in SeqIO.parse(fpath, "fasta"):
                rec_seqs.append(str(rec.seq))
            raw = "".join(rec_seqs)
            clean = _clean_protein(raw, test_mode=test_mode)
            key = os.path.splitext(fname)[0]  # drop extension
            data[key] = {"base_pairs": clean}
            if seq_report:
                logger.info(f"[{strn_or_phg}] {fname}: {len(clean)} aa after cleaning")
        except Exception as e:
            logger.error(f"Failed parsing {fpath}: {e}")

    return data

def save_to_dir(path: str, embeddings, pads, strn_or_phage: str):
    """
    Save embeddings + pad info.
    """
    os.makedirs(path, exist_ok=True)
    torch.save(embeddings, os.path.join(path, f"{strn_or_phage}_embeddings.pt"))
    torch.save(pads, os.path.join(path, f"{strn_or_phage}_pads.pt"))

def complete_n_select(data_dict: Dict[str, Dict[str, str]], context_len: int) -> Tuple[Dict[str, list], Dict[str, list]]:
    """
    Chunk sequences into plain strings of length <= context_len.
    (Strings are later turned into 'M A D E ...' for ProtT5; ESM2 gets raw strings.)
    Returns:
      - chunked: {key: [chunk_str, ...]}
      - pad_info: {key: [chunk_len, ...]}
    """
    chunked = {}
    pad_info = {}
    for key, val in data_dict.items():
        seq = val["base_pairs"]
        if not seq:
            chunked[key] = []
            pad_info[key] = []
            continue
        chunks = [seq[i:i + context_len] for i in range(0, len(seq), context_len)]
        chunked[key] = chunks
        pad_info[key] = [len(c) for c in chunks]
    return chunked, pad_info
