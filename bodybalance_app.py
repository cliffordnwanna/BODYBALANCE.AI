import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import streamlit as st
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

# Define the file URL
url = 'https://drive.google.com/uc?id=17oFQy97Loft7KY1EE5odtPMo2nCZCmzY&export=download'

# Define the local file path to save the downloaded file
local_file_path = 'training_data.txt'

# Download the file from Google Drive
def download_file(url, local_file_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)

# Check if the file has been downloaded successfully
try:
    download_file(url, local_file_path)
    print("File downloaded successfully!")
    print("File saved at:", os.path.abspath(local_file_path))  # Print the absolute path of the saved file
except Exception as e:
    print("Failed to download the file:", str(e))

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load stopwords corpus
stop_words = set(stopwords.words('english'))

# Load the questions and answers from the text file
qa_pairs = {}
with open(local_file_path, 'r') as file:
    lines = file.readlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("question:"):
            question = lines[i].strip().split(':', 1)[1].strip()  # Split at the first occurrence of ":"
            i += 1
            if i < len(lines) and lines[i].strip().startswith("answer:"):
                answer = lines[i].strip().split(':', 1)[1].strip()  # Split at the first occurrence of ":"
                qa_pairs[question] = answer
            else:
                st.warning(f"Invalid format for question: {question}")
        i += 1

# Function to preprocess text
def preprocess_text(text):
    # Tokenize text into words
    tokens = word_tokenize(text.lower())
    # Remove punctuation and stopwords
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

# Function to train Word2Vec model
def train_word2vec_model(sentences):
    # Train the Word2Vec model
    if sentences:
        word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)
        print("Word2Vec model trained successfully!")
    else:
        print("No valid sentences found. Cannot train Word2Vec model.")
    return word2vec_model

# Function to calculate cosine similarity
def calculate_cosine_similarity(user_input, questions):
    vectorizer = TfidfVectorizer(tokenizer=preprocess_text)
    tfidf_matrix = vectorizer.fit_transform([user_input] + questions)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarities

# Function to find the most similar question
def find_similar_question(user_input):
    user_tokens = preprocess_text(user_input)
    questions = list(qa_pairs.keys())
    
    # Check if either user input or questions list is empty
    if not user_tokens or not questions:
        return None
    
    similarities = calculate_cosine_similarity(user_input, questions)
    max_sim_idx = similarities.argmax()
    max_similarity = similarities[max_sim_idx]
    
    if max_similarity > 0:
        most_similar_question = questions[max_sim_idx]
        return most_similar_question
    else:
        return None

# Streamlit app
def main():
    st.title("BODYBALANCE.AI")
    st.write("Hello! I'm a chatbot designed by Clifford.")
    st.write("How can I help you today ?")

    st.write("You can also choose from the options below:")
    st.write("About BodyBalance| Product Information | Product catalog | Ordering Process | Shipping and Delivery | Return Policy | Technical Support | Contact and Assistance | Special Offers and Promotions")

    input_mode = st.radio("Select Input Mode:", ("Text", "Speech"))

    if input_mode == "Text":
        user_input = st.text_input("User:")
        if st.button("Submit"):
            similar_question = find_similar_question(user_input)
            if similar_question:
                st.write("Chatbot:", qa_pairs[similar_question])
            else:
                st.write("Chatbot: Sorry, I couldn't find a relevant answer. Please try rephrasing your question.")
    else:  # Speech input
        st.write("Speech input is not yet supported.")

if __name__ == "__main__":
    main()
