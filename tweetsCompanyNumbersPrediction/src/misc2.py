from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load your pre-trained BERTopic model
# topic_model = BERTopic.load("path_to_your_model")

# Initialize the transformer model for embeddings (use the same as BERTopic's)
model = SentenceTransformer('all-MiniLM-L6-v2')

# The word you are interested in
word = "environment"

# Get the embedding for the given word
word_embedding = model.encode([word])

# Get the top N words for each topic
top_n_words = topic_model.get_topic_freq().head(10)

# Calculate the average embedding for the top N words in each topic
topic_embeddings = []
for topic in top_n_words['Topic']:
    if topic != -1:  # Exclude the outlier topic
        words = [word[0] for word in topic_model.get_topic(topic)]
        word_embeddings = model.encode(words)
        topic_embeddings.append(np.mean(word_embeddings, axis=0))

# Calculate cosine similarity
similarities = cosine_similarity(word_embedding, topic_embeddings)

# Rank topics based on similarity
most_similar_topics = np.argsort(similarities[0])[::-1]

# Print most similar topics
print("Most similar topics to the word 'environment':")
for topic in most_similar_topics[:5]:  # Top 5 similar topics
    print(f"Topic {topic}: {topic_model.get_topic(topic)}")