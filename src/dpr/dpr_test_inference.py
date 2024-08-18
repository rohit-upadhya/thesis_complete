import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer

# Load fine-tuned models
question_encoder = DPRQuestionEncoder.from_pretrained("output/dpr_models/dpr_question_encoder").to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
context_encoder = DPRContextEncoder.from_pretrained("output/dpr_models/dpr_context_encoder").to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Load tokenizers
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# Example inference
question = "What is the capital of France?"
passage = "Paris is the capital of France."

question_inputs = question_tokenizer(question, return_tensors='pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
passage_inputs = context_tokenizer(passage, return_tensors='pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

with torch.no_grad():
    question_embedding = question_encoder(**question_inputs).pooler_output
    passage_embedding = context_encoder(**passage_inputs).pooler_output

# Compute similarity score
similarity_score = torch.matmul(question_embedding, passage_embedding.T)
print("Similarity score:", similarity_score.item())
