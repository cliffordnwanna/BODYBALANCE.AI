import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import streamlit as st
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
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
    try:
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
    except FileNotFoundError:
        st.error("FAQ file not found.")
    return qa_pairs

# Text preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

# Cosine similarity calculation
def calculate_cosine_similarity(user_input, questions):
    vectorizer = TfidfVectorizer(tokenizer=preprocess_text, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform([user_input] + questions)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarities

# Find the most similar question
def find_similar_question(user_input, qa_pairs, threshold=0.3):
    questions = list(qa_pairs.keys())
    if not questions:
        return None
    similarities = calculate_cosine_similarity(user_input, questions)
    max_sim_idx = similarities.argmax()
    max_similarity = similarities[max_sim_idx]
    if max_similarity > threshold:
        return questions[max_sim_idx]
    return None

# Streamlit app
def main():
    st.title("BODYBALANCE.AI")
    st.write("Hello! I'm a chatbot designed by Clifford.")
    st.write("How can I help you today?")

    # Download FAQ file
    url = 'https://drive.google.com/uc?id=17oFQy97Loft7KY1EE5odtPMo2nCZCmzY&export=download'
    local_file_path = 'training_data.txt'
    if not os.path.exists(local_file_path):
        success, message = download_file(url, local_file_path)
        if not success:
            st.error(f"Failed to download the FAQ file: {message}")
            return

    # Load QA pairs
    qa_pairs = load_qa_pairs(local_file_path)
    if not qa_pairs:
        st.warning("No questions and answers loaded. Please check the FAQ file.")
        return

    # User input
    user_input = st.text_input("Ask a question:")
    threshold = st.slider("Set similarity threshold", 0.1, 1.0, 0.3)
    if st.button("Submit"):
        if not user_input.strip():
            st.warning("Please enter a question.")
        elif len(user_input.strip()) < 3:
            st.warning("Input too short. Please provide more details.")
        else:
            similar_question = find_similar_question(user_input, qa_pairs, threshold)
            if similar_question:
                st.success(f"Chatbot: {qa_pairs[similar_question]}")
            else:
                st.warning("Sorry, I couldn't find a relevant answer. Please try rephrasing.")

if __name__ == "__main__":
    main()
