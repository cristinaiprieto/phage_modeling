#!/usr/bin/env python3
"""
SLURM workflow submission script for protein embeddings (ProtT5 / ESM2).
Run: python3 submit_embeddings.py
"""

import subprocess
import sys
import argparse


def main():
    
    parser = argparse.ArgumentParser(description="Run SLURM submission script.")
    parser.add_argument('--model_name', type=str, default='ProtT5', help='Model name to use, e.g. ProtT5 or ESM2')
    args = parser.parse_args()
    model_name = args.model_name
    phage_input_fasta = "/global/home/users/ciprietowitvoet/pLM/phage_modeling/pLMs/proteins/phage_AAs"
    strain_input_fasta = "/global/home/users/ciprietowitvoet/pLM/phage_modeling/pLMs/proteins/strain_AAs"
    strain_output_dir = "/global/home/users/ciprietowitvoet/pLM/phage_modeling/pLMs/output_embeddings/strain"
    phage_output_dir = "/global/home/users/ciprietowitvoet/pLM/phage_modeling/pLMs/output_embeddings/phage"
    root_dir = "/global/home/users/ciprietowitvoet/pLM/phage_modeling"

    # =============================================
    # SLURM CONFIGURATION
    # =============================================
    account = "ac_mak"
    partition = "es1" 
    qos = "es_normal"  
    gpu = "gpu:H100:1" 

    # =============================================
    # WORKFLOW PARAMETERS
    # =============================================
    script_name = "slurm_script.py" 
    model_name = model_name
    batch_size = "8"
    dry_run = False  # Set True to only create scripts but not submit

    # =============================================
    # BUILD COMMAND
    # =============================================
    cmd = [
        "python3", script_name,
        
        # Required arguments for embedding script
        "--phage_in", phage_input_fasta,
        "--strain_in", strain_input_fasta,
        "--strain_out", strain_output_dir,
        "--phage_out", phage_output_dir,
        "--model_name", model_name,
        # "--batch_size", batch_size,
        
        # SLURM / environment configs
        "--account", account,
        "--partition", partition,
        "--qos", qos,
        "--root_dir", root_dir,
        "--gpu", gpu,
    ]

    if dry_run:
        cmd.append("--dry_run")

    # =============================================
    # SUBMIT WORKFLOW
    # =============================================
    print("=" * 60)
    print("SLURM Workflow Submission for Protein Embeddings")
    print("=" * 60)
    print(f"SLURM account:     {account}")
    print(f"Partition:         {partition}")
    print(f"QoS:               {qos}")
    print(f"GPU:               {gpu}")
    # print()
    # print(f"Input fasta:       {input_fasta}")
    # print(f"Output directory:  {output_dir}")
    print(f"Model name:        {model_name}")
    # print()

    if dry_run:
        print("DRY RUN MODE - Scripts will be created but not submitted")
        print()

    print("Submitting workflow with command:")
    print(" ".join(cmd))
    print()

    try:
        subprocess.run(cmd, check=True)
        if dry_run:
            print("Dry run completed successfully! Scripts created.")
        else:
            print("Workflow submitted successfully!")
            print("Monitor progress with:")
            print("  squeue -u $USER")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting workflow: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
