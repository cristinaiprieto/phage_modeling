#!/usr/bin/env python3
"""
SLURM workflow submission script for protein embeddings (ProtT5 / ESM2).
Edit the paths and parameters below, then run: python3 submit_embeddings.py
"""

import subprocess
import sys

def main():
    # =============================================
    # YOUR PATHS - EDIT THESE
    # =============================================
    input_fasta = "/usr2/people/cristinaprieto/phage_modeling/genome_AAs/all_sequences.fasta"  # Your merged fasta file
    output_dir = "/usr2/people/cristinaprieto/phage_modeling/pLMs/embeddings/combined_workflow"
    root_dir = "/usr2/people/cristinaprieto/phage_modeling"

    # =============================================
    # SLURM CONFIGURATION
    # =============================================
    account = "ac_mak"
    partition = "es1"                    # SLURM partition
    qos = "es_normal"                   # SLURM QOS
    gpu = "gpu:1"                      # GPU resource

    # =============================================
    # WORKFLOW PARAMETERS
    # =============================================
    script_name = "main.py"  # Your embedding script filename
    model_name = "ProtT5"    # or "ESM2"
    batch_size = "8"
    dry_run = False          # Set True to only create scripts but not submit

    # =============================================
    # BUILD COMMAND
    # =============================================
    cmd = [
        "python3", script_name,
        
        # Required arguments for your embedding script
        "--input_fasta", input_fasta,
        "--output_path", output_dir,
        "--model_name", model_name,
        "--batch_size", batch_size,
        
        # SLURM / environment configs if needed
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
    print()
    print(f"Input fasta:       {input_fasta}")
    print(f"Output directory:  {output_dir}")
    print(f"Model name:        {model_name}")
    print()

    if dry_run:
        print("🧪 DRY RUN MODE - Scripts will be created but not submitted")
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
            print("  tail -f slurm_run_*/logs/*.out")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting workflow: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
