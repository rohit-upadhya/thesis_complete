# from bertopic import BERTopic
# from typing import List
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer

# class TopicModeling:
#     def __init__(self) -> None:
#         pass

#     def obtain_topic_embeddings(
#         self, 
#         embeddings: np.ndarray,
#         paragraphs: List[str]
#     ) -> np.ndarray:
        
#         if not isinstance(embeddings, np.ndarray):
#             embeddings = embeddings.cpu().detach().numpy()
            
#         vectorizer_model = CountVectorizer(stop_words="english")
#         # min_topic_size = min(len(paragraphs), 10)
        
#         if len(paragraphs) < 10:
#             min_topic_size = min(len(paragraphs), 10)
#             topic_model = BERTopic(
#                 vectorizer_model=vectorizer_model,
#                 min_topic_size=min_topic_size,
#                 nr_topics=25,
#                 calculate_probabilities=True,
#                 low_memory=False
#             )
#         else:
#             topic_model = BERTopic(
#                 vectorizer_model=vectorizer_model,
#                 # min_topic_size=min_topic_size,
#                 nr_topics=25,
#                 calculate_probabilities=True,
#                 low_memory=False
#             )
        
        
#         topic, probabilities = topic_model.fit_transform(paragraphs, embeddings=embeddings)
#         topic_embeddings = topic_model.topic_embeddings_[1:]
        
#         # if probabilities is None or probabilities.ndim == 1 or probabilities.size == 0:
#         #     probabilities = np.ones((len(paragraphs), 1))
#         #     topic_embeddings = topic_embeddings[:1]
        
#         # else:
#         #     noise_probabilities = 1 - probabilities.sum(axis=1, keepdims=True)
#         #     probabilities = np.hstack([noise_probabilities, probabilities])
        
        
#         return probabilities, topic_embeddings


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
    def __init__(self, model_name: str = "all-mpnet-base-v2") -> None:
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_model.eval()

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
