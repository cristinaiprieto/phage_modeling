import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel
from utils import save_to_dir, rt_dicts, complete_n_select
import logging
import esm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model loader

def get_model_and_tokenizer(model_name):
    """
    Load a pretrained pLM and its tokenizer/converter.
    For ProtT5: return (HF tokenizer, HF model)
    For ESM2:  return (batch_converter, ESM model)  [Meta implementation]
    """
    if model_name == "ProtT5":
        logger.info("ProtT5 model selected.")
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").eval()
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        logger.info("Tokenizer ready.")
        return tokenizer, model

    elif model_name == "ESM2":
        logger.info("ESM2 model (Meta implementation) selected.")
        model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t6_8M_UR50D")
        model.eval()
        batch_converter = alphabet.get_batch_converter()
        logger.info("Batch converter ready.")
        return batch_converter, model

    else:
        raise ValueError(f"Model {model_name} not supported. Choose 'ProtT5' or 'ESM2'.")

# ProtT5 helpers (unchanged)

class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return {"base_pairs": self.sequences[idx]}

def string_collate_fn(batch):
    return [item["base_pairs"] for item in batch]

def tokenize_protein_sequences(tokenizer, sequences, max_length=1024):
    # HuggingFace tokenizer
    return tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

# Embedding extraction

def extract_embeddings(sequences, context_len, tokenize_func, model, model_name, batch_size=8, test_mode=False):
    """
    Extract embeddings for a list of protein sequence strings.
    - ProtT5 path: HuggingFace tokenizer + mean pool over non-pad tokens
    - ESM2 path:  Meta ESM batch_converter + mean over residues (skip special tokens)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    all_embeddings = []

    if model_name == "ESM2":
        # Use Meta ESM batch converter; tokenize_func is the batch_converter callable
        batch_converter = tokenize_func  # callable that accepts list[(label, seq)]
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                sub = sequences[start:start + batch_size]
                # labels can just be indices; ESM does not use them beyond identification
                batch = [(str(start + i), s) for i, s in enumerate(sub)]
                labels, strs, toks = batch_converter(batch)
                toks = toks.to(device, non_blocking=True)

                out = model(toks, repr_layers=[model.num_layers])
                reps = out["representations"][model.num_layers]  # (B, L, D)

                # Mean pool across residues (exclude special tokens: index 0 is BOS)
                for i, s in enumerate(strs):
                    L = min(len(s), context_len)  # respect truncation length
                    # residues from 1..L inclusive (skip BOS)
                    emb = reps[i, 1:L + 1].mean(0).detach().cpu().numpy()
                    all_embeddings.append(emb)

        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, model.embed_dim))

    else:
        # ProtT5 (HuggingFace)
        dataset = SequenceDataset(sequences)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=string_collate_fn)

        with torch.no_grad():
            for batch in dataloader:
                # ProtT5 expects space-separated characters
                spaced = [" ".join(seq) for seq in batch]
                tokenized = tokenize_protein_sequences(tokenize_func, spaced, max_length=context_len)
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
                logger.info(f"ProtT5 batch size: {len(batch)}")

                outputs = model(**tokenized)
                last_hidden = outputs.last_hidden_state  # (B, L, D)
                logger.info(f"ProtT5 last_hidden: {tuple(last_hidden.shape)}")

                # Mean-pool over non-padding tokens
                input_mask = tokenized["attention_mask"]  # (B, L)
                sum_embeddings = (last_hidden * input_mask.unsqueeze(-1)).sum(dim=1)
                lengths = input_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
                pooled = (sum_embeddings / lengths).detach().cpu().numpy()
                all_embeddings.append(pooled)

        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, model.config.d_model))

# Workflow

def embedding_workflow(model_name, context_len, strain_in, strain_out, phage_in, phage_out, early_exit=False, test_mode=False):
    """
    Runs the pLM workflow to extract embeddings.
    """
    ecoli_strains = rt_dicts(path=strain_in, seq_report=True, test_mode=test_mode)
    ecoli_phages  = rt_dicts(path=phage_in,  strn_or_phg='phage', seq_report=True, test_mode=test_mode)
    logger.info(f"Loaded {len(ecoli_strains)} strains, {len(ecoli_phages)} phages")

    if early_exit:
        logger.info("Early exit triggered.")
        return

    tokenizer_or_converter, model = get_model_and_tokenizer(model_name)
    logger.info('Model + tokenizer/converter ready')

    # Chunk into <= context_len pieces (strings)
    logger.info("Chunking input sequences")
    estrain_n_select, estrain_pads = complete_n_select(ecoli_strains, context_len)
    ephage_n_select,  ephage_pads  = complete_n_select(ecoli_phages,  context_len)

    # Flatten chunks to a single list for each group
    strain_chunks = [chunk for chunks in estrain_n_select.values() for chunk in chunks]
    phage_chunks  = [chunk for chunks in ephage_n_select.values()  for chunk in chunks]

    if model_name == "ESM2":
        # For ESM2, tokenize_func is the batch_converter callable returned above
        tokenize_func = tokenizer_or_converter
    else:
        # For ProtT5, tokenize_func is the HF tokenizer
        tokenize_func = tokenizer_or_converter

    logger.info(f"Extracting strain embeddings from {len(strain_chunks)} chunks")
    estrain_embed = extract_embeddings(
        strain_chunks, context_len, tokenize_func, model, model_name, test_mode=test_mode
    )

    logger.info(f"Extracting phage embeddings from {len(phage_chunks)} chunks")
    ephage_embed = extract_embeddings(
        phage_chunks, context_len, tokenize_func, model, model_name, test_mode=test_mode
    )

    logger.info(f"Saving embeddings: strain {estrain_embed.shape}, phage {ephage_embed.shape}")
    save_to_dir(strain_out, embeddings=estrain_embed, pads=estrain_pads, strn_or_phage='strain')
    save_to_dir(phage_out,  embeddings=ephage_embed,  pads=ephage_pads,  strn_or_phage='phage')

    logger.info("Embedding workflow complete.")
