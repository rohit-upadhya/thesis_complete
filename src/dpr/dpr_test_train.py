import json
from datasets import Dataset
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer, AdamW
import torch
from torch.utils.data import DataLoader

with open("raw_data/test/train_data.json", "r") as file:
    data = json.load(file)

dataset = Dataset.from_list(data)

def preprocess_function(examples):
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    
    questions = question_tokenizer(examples['question_text'], truncation=True, padding="max_length", max_length=64, return_tensors='pt')
    passages = context_tokenizer(examples['passage_text'], truncation=True, padding="max_length", max_length=256, return_tensors='pt')

    return {
        'input_ids_question': questions['input_ids'],
        'attention_mask_question': questions['attention_mask'],
        'input_ids_passage': passages['input_ids'],
        'attention_mask_passage': passages['attention_mask'],
        'label': examples['label']
    }

encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset.set_format(type='torch', columns=['input_ids_question', 'attention_mask_question', 'input_ids_passage', 'attention_mask_passage', 'label'])

train_dataloader = DataLoader(encoded_dataset, batch_size=8, shuffle=True)

# Load models and optimizer
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

optimizer = AdamW(list(question_encoder.parameters()) + list(context_encoder.parameters()), lr=2e-5)

# Training loop
num_epochs = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(num_epochs):
    question_encoder.train()
    context_encoder.train()
    
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        question_outputs = question_encoder(input_ids=batch['input_ids_question'], attention_mask=batch['attention_mask_question']).pooler_output
        passage_outputs = context_encoder(input_ids=batch['input_ids_passage'], attention_mask=batch['attention_mask_passage']).pooler_output

        similarity_scores = torch.matmul(question_outputs, passage_outputs.T)

        labels = torch.arange(similarity_scores.size(0)).long().to(similarity_scores.device)
        loss = torch.nn.CrossEntropyLoss()(similarity_scores, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

question_encoder.save_pretrained("output/dpr_models/dpr_question_encoder")
context_encoder.save_pretrained("output/dpr_models/dpr_context_encoder")
