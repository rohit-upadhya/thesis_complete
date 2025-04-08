import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer

question_encoder = DPRQuestionEncoder.from_pretrained("output/dpr_models/dpr_question_encoder").to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
context_encoder = DPRContextEncoder.from_pretrained("output/dpr_models/dpr_context_encoder").to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

passages = [
    "Paris is the capital of France.",
    "'Pride and Prejudice' was written by Jane Austen.",
    "The largest planet in our solar system is Jupiter.",
    "Water boils at 100 degrees Celsius.",
    "The Mona Lisa was painted by Leonardo da Vinci."
]

def find_most_relevant_passage(question, passages):
    question_inputs = question_tokenizer(question, return_tensors='pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    with torch.no_grad():
        question_embedding = question_encoder(**question_inputs).pooler_output

    best_passage = None
    highest_score = float('-inf')

    for passage in passages:
        passage_inputs = context_tokenizer(passage, return_tensors='pt').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        with torch.no_grad():
            passage_embedding = context_encoder(**passage_inputs).pooler_output

        similarity_score = torch.matmul(question_embedding, passage_embedding.T).item()

        if similarity_score > highest_score:
            highest_score = similarity_score
            best_passage = passage

    return best_passage, highest_score

question = "Who wrote 'Pride and Prejudice'?"

answer, score = find_most_relevant_passage(question, passages)
print(f"Question: {question}")
print(f"Answer: {answer}")
print(f"Similarity score: {score}")
