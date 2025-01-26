from bertopic import BERTopic
from typing import List
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class TopicModeling:
    def __init__(self) -> None:
        self.count = 0
        pass

    def obtain_topic_embeddings(
        self, 
        embeddings: np.ndarray,
        paragraphs: List[str]
    ) -> np.ndarray:
        
        if not isinstance(embeddings, np.ndarray):
            embeddings = embeddings.cpu().detach().numpy()
            
        vectorizer_model = CountVectorizer(stop_words="english")
        
        # if len(paragraphs) < 10:
        min_topic_size = min(len(paragraphs), 4)
        topic_model = BERTopic(
            vectorizer_model=vectorizer_model,
            min_topic_size=min_topic_size,
            nr_topics=25,
            calculate_probabilities=True,
            low_memory=False
        )
        # else:
        #     topic_model = BERTopic(
        #         vectorizer_model=vectorizer_model,
        #         nr_topics=25,
        #         calculate_probabilities=True,
        #         low_memory=False
        #     )
        
        
        topic, probabilities = topic_model.fit_transform(paragraphs, embeddings=embeddings)
        topic_embeddings = topic_model.topic_embeddings_[1:]
        if len(topic_embeddings) == 0:
            self.count += 1
        return probabilities, topic_embeddings
        # print("zero noticed")
        # return np.expand_dims(np.ones(len(paragraphs)), axis=1), topic_model.topic_embeddings_