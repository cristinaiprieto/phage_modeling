#!/usr/bin/env python3
"""
plMs SLURM Workflow. Writes a SLURM batch script
"""

import os
import argparse
import subprocess
import time
from datetime import datetime

class ProteinPipelineConfig():
    def __init__(self, args):
        self.args = args
        strain_output_dir = args.strain_output_path
        phage_output_dir = args.phage_output_path
        self.completion_markers = {
            1: os.path.join(phage_output_dir, "phage_embedding_complete.txt"),
            2: os.path.join(strain_output_dir, "strain_embedding_complete.txt")
        }
        self.stage_names = {
            1: "Embedding", 
        }

    def check_stage_completion(self, stage):
        marker = self.completion_markers.get(stage)
        if marker and os.path.exists(marker):
            print(f"Stage {stage} appears complete (found: {marker})")
            return True
        return False

def submit_job(script_path, dependency=None):
    cmd = ['sbatch', '--parsable']
    if dependency:
        cmd += ['--dependency', f'afterok:{dependency}']
    cmd.append(script_path)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error submitting {script_path}: {e}")
        print(f"Error output: {e.stderr}")
        return None

def create_embedding_script(args, run_dir):
    script = f"""#!/bin/bash
#SBATCH --job-name=protein_embedding
#SBATCH --account={args.account}
#SBATCH --partition={args.partition}
#SBATCH --qos={args.qos}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres={args.gpu}
#SBATCH --time=2:00:00
#SBATCH --output=logs/embedding_%j.out
#SBATCH --error=logs/embedding_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"

module load ml/pytorch
module load anaconda3
conda activate {"phage_modeling_env"} 2>&1 || {{
    echo "Direct activation failed, trying with conda init..."
    conda init bash >/dev/null 2>&1
    source ~/.bashrc >/dev/null 2>&1
    conda activate {"phage_modeling_env"}
}}

cd {args.root_dir}
pip install -e .

echo "Running embedding script..."
python3 {args.script} \\
    --strain_input_fasta {args.strain_input_fasta} \\
    --phage_input_fasta {args.phage_input_fasta} \\
    --output_path {args.output_path} \\
    --model_name {args.model_name} \\
    --batch_size {args.batch_size}

touch {args.strain_output_path}/strain_embedding_complete.txt
touch {args.phage_output_path}/phage_embedding_complete.txt
echo "Completed: $ (date)"
"""
    path = os.path.join(run_dir, "main.sh")
    with open(path, 'w') as f:
        f.write(script)
    os.chmod(path, 0o755)
    return path

def main():
    parser = argparse.ArgumentParser(description="Submit protein embedding SLURM job")
    parser.add_argument('--strain_input_fasta', required=True)
    parser.add_argument('--phage_input_fasta', required=True)
    parser.add_argument('--strain_output_path', required=True)
    parser.add_argument('--phage_output_path', required=True)

    # parser.add_argument('--output_npz', required=True)
    # parser.add_argument('--output', required=True)
    # see ESM2_embeddings_by_dir

    parser.add_argument('--model_name', type=str, default='ProtT5', help='Name of the model (e.g., ProtT5, ESM2)')
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--account', default='ac_mak')
    parser.add_argument('--partition', default='es1')
    parser.add_argument('--qos', default='es_normal')
    parser.add_argument('--gpu', default='gpu:1')
    parser.add_argument('--root_dir', default='.')
    parser.add_argument('--script', default='protembed/main.py')
    parser.add_argument('--dry_run', action='store_true')

    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = f"slurm_embed_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    config = ProteinPipelineConfig(args)

    print(f"Creating SLURM script in: {run_dir}")
    embed_script = create_embedding_script(args, run_dir)

    if args.dry_run:
        print("Dry run: SLURM script created but not submitted.")
        return

    os.chdir(run_dir)

    if not config.check_stage_completion(1):
        job_id = submit_job("main.sh")
        print(f"Submitted embedding job with ID: {job_id}")
    else:
        print("Embedding stage already complete.")

    print(f"\nCheck logs in {os.path.join(run_dir, 'logs')}")

if __name__ == "__main__":
    main()
