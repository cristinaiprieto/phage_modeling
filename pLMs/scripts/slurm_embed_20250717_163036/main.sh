#!/bin/bash
#SBATCH --job-name=protein_embedding
#SBATCH --account=ac_mak
#SBATCH --partition=es1
#SBATCH --qos=es_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:H100:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/embedding_%j.out
#SBATCH --error=logs/embedding_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"

module load ml/pytorch
module load anaconda3
conda activate phage_modeling_env 2>&1 || {
    echo "Direct activation failed, trying with conda init..."
    conda init bash >/dev/null 2>&1
    source ~/.bashrc >/dev/null 2>&1
    conda activate phage_modeling_env
}

cd /global/home/users/ciprietowitvoet/pLM/phage_modeling
pip install -e .

echo "Running embedding script..."
python3 protembed/main.py \
    --strain_input_fasta /global/home/users/ciprietowitvoet/pLM/phage_modeling/pLMs/proteins/strain_AAs \
    --phage_input_fasta /global/home/users/ciprietowitvoet/pLM/phage_modeling/pLMs/proteins/phage_AAs \
    --output_path /global/home/users/ciprietowitvoet/pLM/phage_modeling/pLMs/output_embeddings \
    --model_name ProtT5 \
    --batch_size 8

touch /global/home/users/ciprietowitvoet/pLM/phage_modeling/pLMs/output_embeddings/embedding_complete.txt
echo "Completed: $ (date)"
