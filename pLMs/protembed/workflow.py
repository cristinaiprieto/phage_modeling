import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np 
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModel
from utils import save_to_dir, rt_dicts, complete_n_select
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_model_and_tokenizer(model_name):
    """
    Load a pretrained pLM and its tokenizer.
    """
    if model_name == "ProtT5":
        logger.info("ProtT5 model selected.")
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").eval()
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        logger.info("Tokenizer ready.")
    elif model_name == "ESM2":
        esm_model_id = "facebook/esm2_t6_8M_UR50D"  # or swap for a larger one
        logger.info("ESM2 model selected.")
        tokenizer = AutoTokenizer.from_pretrained(esm_model_id, do_lower_case=False)
        model = AutoModel.from_pretrained(esm_model_id).eval()
        logger.info("Tokenizer ready.")
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

class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"base_pairs": self.sequences[idx]}
    
def string_collate_fn(batch):
    return [item["base_pairs"] for item in batch]  

def extract_embeddings(sequences, context_len, tokenize_func, model, model_name, batch_size=8, test_mode=False):
    """
    Extract embeddings from a list of protein sequences using a pretrained model.
    Returns a list of mean pooled embeddings (1 per sequence).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = SequenceDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=string_collate_fn)

    all_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            tokenize_func(batch)
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            outputs = model(**tokenized)
            last_hidden = outputs.last_hidden_state

            if model_name == "ESM2":
                # Use [CLS] token (token at index 0)
                pooled = last_hidden[:, 0, :].cpu().numpy()
            else:
                # Mean-pool over non-padding tokens for ProtT5
                input_mask = tokenized["attention_mask"]
                sum_embeddings = (last_hidden * input_mask.unsqueeze(-1)).sum(dim=1)
                lengths = input_mask.sum(dim=1).unsqueeze(-1)
                pooled = (sum_embeddings / lengths).cpu().numpy()

            all_embeddings.append(pooled)

    return np.vstack(all_embeddings)

def embedding_workflow(model_name, context_len, strain_in, strain_out, phage_in, phage_out, early_exit=False, test_mode=False):
    """
    Runs the pLM workflow to extract embeddings.
    """
    ecoli_strains = rt_dicts(path=strain_in, seq_report=True, test_mode=test_mode)
    ecoli_phages = rt_dicts(path=phage_in, strn_or_phg='phage', seq_report=True, test_mode=test_mode)

    logger.info('ecoli strains and phages defined')

    if early_exit:
        logger.info("Early exit triggered.")
        return

    tokenizer, model = get_model_and_tokenizer(model_name)
    logger.info('tokenizer and model defined')

    def tokenize_func(batch, max_length=context_len):
        if model_name == "ProtT5":
            batch = [" ".join(seq) for seq in batch]
        return tokenize_protein_sequences(tokenizer, batch, max_length=max_length)
    
    logger.info("Chunking input sequences")
    estrain_n_select, estrain_pads = complete_n_select(ecoli_strains, context_len)
    ephage_n_select, ephage_pads = complete_n_select(ecoli_phages, context_len)

    logger.info("Extracting strain embeddings")
    estrain_embed = extract_embeddings(list(estrain_n_select.values()), context_len, tokenize_func, model, model_name, test_mode=test_mode)

    logger.info("Extracting phage embeddings")
    ephage_embed = extract_embeddings(list(ephage_n_select.values()), context_len, tokenize_func, model, model_name, test_mode=test_mode)

    logger.info("Saving embeddings to output directories")
    save_to_dir(strain_out, embeddings=estrain_embed, pads=estrain_pads, strn_or_phage='strain')
    save_to_dir(phage_out, embeddings=ephage_embed, pads=ephage_pads, strn_or_phage='phage')

    logger.info("Embedding workflow complete.")