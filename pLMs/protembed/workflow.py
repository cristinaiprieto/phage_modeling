import torch
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModel
from .utils import save_to_dir, rt_dicts, complete_n_select
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_model_and_tokenizer(model_name):
    """
    Load a pretrained pLM and its tokenizer.
    """
    if model_name == "ProtT5":
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").eval()
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    elif model_name == "ESM2":
        esm_model_id = "facebook/esm2_t6_8M_UR50D"  # or swap for a larger one
        tokenizer = AutoTokenizer.from_pretrained(esm_model_id, do_lower_case=False)
        model = AutoModel.from_pretrained(esm_model_id).eval()
    else:
        raise ValueError(f"Model {model_name} not supported. Choose 'ProtT5' or 'ESM2'.")

    return tokenizer, model

def tokenize_protein_sequences(tokenizer, sequences, max_length=1024):
    """
    Tokenize protein sequences.
    """
    return tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

def embedding_workflow(model_name, context_len, strain_in, strain_out, phage_in, phage_out, bacteria='ecoli', early_exit=False, test_mode=False):
    """
    Runs the pLM workflow to extract embeddings.
    """
    ecoli_strains = rt_dicts(path=strain_in, seq_report=True, test_mode=test_mode)
    ecoli_phages = rt_dicts(path=phage_in, strn_or_phg='phage', seq_report=True, test_mode=test_mode)

    if early_exit:
        logger.info("Early exit triggered.")
        return

    tokenizer, model = get_model_and_tokenizer(model_name)

    def tokenize_func(examples, max_length=context_len):
        return tokenize_protein_sequences(tokenizer, examples["base_pairs"], max_length=max_length)

    logger.info("Chunking input sequences")
    estrain_n_select, estrain_pads = complete_n_select(ecoli_strains, context_len)
    ephage_n_select, ephage_pads = complete_n_select(ecoli_phages, context_len)

    logger.info("Extracting strain embeddings")
    estrain_embed = extract_embeddings(estrain_n_select, context_len, tokenize_func, model, test_mode=test_mode)

    logger.info("Extracting phage embeddings")
    ephage_embed = extract_embeddings(ephage_n_select, context_len, tokenize_func, model, test_mode=test_mode)

    logger.info("Saving embeddings to output directories")
    save_to_dir(strain_out, embeddings=estrain_embed, pads=estrain_pads, name=bacteria, strn_or_phage='strain')
    save_to_dir(phage_out, embeddings=ephage_embed, pads=ephage_pads, name=bacteria, strn_or_phage='phage')

    logger.info("Embedding workflow complete.")