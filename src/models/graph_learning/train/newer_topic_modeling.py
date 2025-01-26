from bertopic import BERTopic
from typing import List, Tuple
import torch
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class TopicModeling:
    def __init__(self, embedding_model: SentenceTransformer) -> None:
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = SentenceTransformer("/srv/upadro/models/graph/new_gat/2025-01-05___all_gat_train_sentence_transformer_threshold_no_weight_0_shot_use_topics_use_prev_next_two_training/checkpoints/epoch_0/topic_model")

    def obtain_topic_embeddings(self, paragraphs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        paragraphs_clean = [clean_text(p) for p in paragraphs]
        
        with torch.no_grad():
            embeddings = self.embedding_model.encode(paragraphs_clean, show_progress_bar=False)
        if not isinstance(embeddings, np.ndarray):
            embeddings = embeddings.cpu().detach().numpy()

        vectorizer_model = CountVectorizer(stop_words="english")

        if len(paragraphs_clean) < 10:
            min_topic_size = min(len(paragraphs_clean), 10)
            topic_model = BERTopic(
                vectorizer_model=vectorizer_model,
                min_topic_size=min_topic_size,
                nr_topics=25,
                calculate_probabilities=True,
                low_memory=False
            )
        else:
            topic_model = BERTopic(
                vectorizer_model=vectorizer_model,
                nr_topics=25,
                calculate_probabilities=True,
                low_memory=False
            )

        topics, probabilities = topic_model.fit_transform(paragraphs_clean, embeddings=embeddings)
        topic_embeddings = topic_model.topic_embeddings_[1:]
        return probabilities, topic_embeddings
