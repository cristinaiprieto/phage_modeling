from transformers import T5EncoderModel, T5Tokenizer
import torch
import os

def generate_prott5_embeddings(input_dir, output_dir):
    # Load tokenizer and ProtT5 model
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    model = model.eval()

    input_dir = '/usr2/people/cristinaprieto/phage_modeling/genome_AAs'
    embeddings = {}

    # Processing each file to add a space in between each amino acid
    for file in os.listdir(input_dir):
        if file.endswith(".faa"):
            filepath = os.path.join(input_dir, file)
            with open(filepath, "r") as f:
                lines = f.readlines()
                seq = "".join([line.strip() for line in lines if not line.startswith(">")])
                spaced_seq = " ".join(seq)

    # Tokenize and get embeddings
    input_ids = tokenizer(spaced_seq, return_tensors="pt").input_ids
    with torch.no_grad():
        embedding = model(input_ids).last_hidden_state.squeeze(0)

    # Save the embedding
    protein_id = os.path.splitext(file)[0]
    output_path = os.path.join(output_dir, f"{protein_id}.pt")
    torch.save(pooled_embedding, output_path)

