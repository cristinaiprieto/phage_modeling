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
        strain_output_dir = args.strain_out
        phage_output_dir = args.phage_out
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
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
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
#SBATCH --mem='80G'
#SBATCH --gres={args.gpu}
#SBATCH --time=12:00:00
#SBATCH --output=logs/embedding_%j.out
#SBATCH --error=logs/embedding_%j.err

export HF_HOME=/global/scratch/users/ciprietowitvoet/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"

SCRATCH_ENV_PATH=/global/scratch/users/$USER/envs/phage_modeling_env

unset MKL_INTERFACE_LAYER
unset MKL_THREADING_LAYER
unset LD_PRELOAD

module load anaconda3
conda activate $SCRATCH_ENV_PATH 2>&1 || {{
    echo "Direct activation failed, trying with conda init..."
    conda init bash >/dev/null 2>&1
    source ~/.bashrc >/dev/null 2>&1
    conda activate $SCRATCH_ENV_PATH
}}

export XDG_CACHE_HOME=/global/scratch/users/$USER/.cache
export TRANSFORMERS_CACHE=$XDG_CACHE_HOME/huggingface
export HF_HOME=$XDG_CACHE_HOME/huggingface
export PIP_CACHE_DIR=$XDG_CACHE_HOME/pip

cd {args.root_dir}

mkdir -p logs
echo "Running embedding script..."
conda run -p $SCRATCH_ENV_PATH python3 {args.script} \\
    --strain_in {args.strain_in} \\
    --phage_in {args.phage_in} \\
    --strain_out {args.strain_out} \\
    --phage_out {args.phage_out} \\
    --model_name {args.model_name} \\

if [ $? -eq 0 ]; then
    echo "Workflow completed successfully."
    touch {args.strain_out}/strain_embedding_complete.txt
    touch {args.phage_out}/phage_embedding_complete.txt
else
    echo "Workflow failed. Completion markers not created."
fi

echo "Completed: $(date)"
"""
    path = os.path.join(run_dir, "main.slurm")
    with open(path, 'w') as f:
        f.write(script)
    os.chmod(path, 0o755)
    return path

def main():
    parser = argparse.ArgumentParser(description="Submit protein embedding SLURM job")
    parser.add_argument('--strain_in', required=True)
    parser.add_argument('--phage_in', required=True)
    parser.add_argument('--strain_out', required=True)
    parser.add_argument('--phage_out', required=True)

    # parser.add_argument('--output_npz', required=True)
    # parser.add_argument('--output', required=True)
    # see ESM2_embeddings_by_dir

    parser.add_argument('--model_name', type=str, default='ProtT5', help='Name of the model (e.g., ProtT5, ESM2)')
    # parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--account', default='ac_mak')
    parser.add_argument('--partition', default='es1')
    parser.add_argument('--qos', default='es_normal')
    parser.add_argument('--gpu', default='gpu:1')
    parser.add_argument('--root_dir', default='.')
    parser.add_argument('--script', default='pLMs/protembed/main.py')
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
        job_id = submit_job("main.slurm")
        print(f"Submitted embedding job with ID: {job_id}")
    else:
        print("Embedding stage already complete.")

    print(f"\nCheck logs in {os.path.join(run_dir, 'logs')}")

if __name__ == "__main__":
    main()
