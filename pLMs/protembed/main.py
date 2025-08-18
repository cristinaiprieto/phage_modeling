import torch
import argparse

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from workflow import embedding_workflow
from esm2_func import run

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

print("Starting main.py script")



def main():

    parser = argparse.ArgumentParser(description="Run protein embedding workflow using ProtT5 or ESM2.")
    parser.add_argument("--model_name", type=str, required=True, choices=["ProtT5", "ESM2"], help="Model name: ProtT5 or ESM2")
    parser.add_argument("--context_len", type=int, default=512, help="Maximum sequence length for tokenization")
    # parser.add_argument("--strain_in", type=str, required=True, help="Path to strain FASTA directory")
    # parser.add_argument("--strain_out", type=str, required=True, help="Output directory for strain embeddings")
    # parser.add_argument("--phage_in", type=str, required=True, help="Path to phage FASTA directory")
    # parser.add_argument("--phage_out", type=str, required=True, help="Output directory for phage embeddings")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory for FASTA files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for embeddings")
    parser.add_argument("--early_exit", action="store_true", help="Flag to skip embedding and exit early")
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode (loads fewer sequences)")

    args = parser.parse_args()

    # embedding_workflow(
    #     model_name=args.model_name,
    #     context_len=args.context_len,
    #     # strain_in=args.strain_in,
    #     # strain_out=args.strain_out,
    #     # phage_in=args.phage_in,
    #     # phage_out=args.phage_out,
    #     input_dir=args.input_dir,
    #     output_dir=args.output_dir,
    #     early_exit=args.early_exit,
    #     test_mode=args.test_mode
    # )

    run(args)

if __name__ == "__main__":
    main()