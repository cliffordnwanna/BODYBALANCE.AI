import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import streamlit as st
import gdown
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Define the file URL
url = 'https://drive.google.com/file/d/17oFQy97Loft7KY1EE5odtPMo2nCZCmzY/view?usp=drive_link'

# Define the local file path to save the downloaded file
local_file_path = 'training_data.txt'

# Download the file from Google Drive
gdown.download(url, local_file_path, quiet=False)

# Check if the file has been downloaded successfully
if os.path.exists(local_file_path):
    print("File downloaded successfully!")
else:
    print("Failed to download the file.")

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

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
