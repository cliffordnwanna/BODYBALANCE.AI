import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import streamlit as st
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to download NLTK resources if not already available
def download_nltk_resources():
    resources = ['punkt', 'stopwords']
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == 'punkt' else resource)
        except LookupError:
            nltk.download(resource)

# Call the function to ensure resources are available
download_nltk_resources()

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# File download utility
def download_file(url, local_file_path):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True, None
    except Exception as e:
        return False, str(e)

# Load QA pairs from the text file
def load_qa_pairs(file_path):
    qa_pairs = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if lines[i].strip().startswith("question:"):
                question = lines[i].strip().split(':', 1)[1].strip()
                i += 1
                if i < len(lines) and lines[i].strip().startswith("answer:"):
                    answer = lines[i].strip().split(':', 1)[1].strip()
                    qa_pairs[question] = answer
            i += 1
    return qa_pairs

# Text preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

# Cosine similarity calculation
def calculate_cosine_similarity(user_input, questions):
    vectorizer = TfidfVectorizer(tokenizer=preprocess_text)
    tfidf_matrix = vectorizer.fit_transform([user_input] + questions)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarities

# Find the most similar question
def find_similar_question(user_input, qa_pairs):
    questions = list(qa_pairs.keys())
    if not questions:
        return None
    similarities = calculate_cosine_similarity(user_input, questions)
    max_sim_idx = similarities.argmax()
    max_similarity = similarities[max_sim_idx]
    if max_similarity > 0.2:  # Threshold to filter irrelevant results
        return questions[max_sim_idx]
    return None

# Streamlit app
def main():
    st.title("BODYBALANCE.AI")
    st.write("Hello! I'm a chatbot designed by Clifford.")
    st.write("How can I help you today?")

    url = 'https://drive.google.com/uc?id=17oFQy97Loft7KY1EE5odtPMo2nCZCmzY&export=download'
    local_file_path = 'training_data.txt'

    # Download FAQ file if not already present
    if not os.path.exists(local_file_path):
        success, message = download_file(url, local_file_path)
        if not success:
            st.error(f"Failed to download the FAQ file: {message}")
            return

    # Load QA pairs
    qa_pairs = load_qa_pairs(local_file_path)

    # User input
    user_input = st.text_input("Ask a question:")
    if st.button("Submit"):
        if not user_input.strip():
            st.warning("Please enter a question.")
        else:
            similar_question = find_similar_question(user_input, qa_pairs)
            if similar_question:
                st.success(f"Chatbot: {qa_pairs[similar_question]}")
            else:
                st.warning("Sorry, I couldn't find a relevant answer. Please try rephrasing.")

if __name__ == "__main__":
    main()
