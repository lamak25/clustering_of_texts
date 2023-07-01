from transformers import BertTokenizer, BertModel
import torch
import json

# BERT model info:
# https://huggingface.co/bert-base-uncased

class embModel:
    def split_sequence(self, sequence, max_length):
        # Split long sequences into bynch of smaller, suitable for the model

        num_splits = len(sequence) // max_length  # Calculate the number of splits required
        splits = []
        for i in range(num_splits):
            start_index = i * max_length
            end_index = (i + 1) * max_length
            split = sequence[start_index:end_index]
            splits.append(split)

        # Add the remaining part of the sequence as the last split
        remaining = sequence[num_splits * max_length:]
        if remaining:
            splits.append(remaining)

        return splits


    def get_tokens_embeddings(self, model, token):
        # Get embeddings from the tokens

        # Convert tokens to a PyTorch tensor (for processing)
        input_tensor = torch.tensor([token])

        # Generate the BERT embeddings
        with torch.no_grad():
            outputs = model(input_tensor)
            embeddings = outputs.last_hidden_state

        # Get the sentence embedding
        return embeddings.squeeze(0).mean(dim=0)

    def get_emb_from_docs(self, all_texts, bert_model_name, max_seq_length):
        # Convert documents into embeddings

        # Load the pre-trained BERT tokenizer and model (uncased = not case-sensitive)
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        model = BertModel.from_pretrained(bert_model_name)
        total_texts = len(all_texts)

        for i, doc in enumerate(all_texts):
            print("Embeddings created: %d/%d (now ID: %d)" % (i+1, total_texts, doc[0]))
            sentence = doc[1]

            # Tokenize the sentence
            tokens = tokenizer.encode(sentence, add_special_tokens=True)

            sentence_embedding = None
            if len(tokens) > max_seq_length:
                # Token is too long fot the model -> split it and join as an average of the resulted embeddings
                print("Token is too long:", len(tokens))
                # Split the sequence into multiple smaller sequences
                split_sequences = self.split_sequence(tokens, max_seq_length)

                split_tensor = []

                # Convert sekvences
                for token in split_sequences:
                    print("Sekvence size:", len(token))
                    split_tensor.append(self.get_tokens_embeddings(model, token))

                # Concatenate the tensors(sekvences) along a new dimension (e.g., dimension 0)
                concatenated_tensor = torch.stack(split_tensor, dim=0)

                # Use & Compute the average to join the tokens
                sentence_embedding = torch.mean(concatenated_tensor, dim=0)
            else:
                sentence_embedding = self.get_tokens_embeddings(model, tokens)
                
            # print(sentence_embedding)
            doc[1] = sentence_embedding.tolist()

        return all_texts

    def save_embeddings(self, emb, file_path):
        # Export the array as JSON into a file

        print("Saving embeddings...")
        with open(file_path, 'w+') as output_file:
            json.dump(emb, output_file)

    def load_embeddings(self, file_path):
        # Load JSON data from the file

        print("Loading embeddings...")
        json_data = None
        with open(file_path, 'r') as input_file:
            json_data = json.load(input_file)
        return json_data
