import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import streamlit as st
import gdown
import os



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
            question = lines[i].strip().split(':')[1].strip()
            i += 1
            if i < len(lines) and lines[i].strip().startswith("answer:"):
                answer = lines[i].strip().split(':')[1].strip()
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

# Function to find the most similar question
def find_similar_question(user_question):
    user_tokens = preprocess_text(user_question)
    max_sim = -1
    most_similar_question = None
    for question, answer in qa_pairs.items():
        question_tokens = preprocess_text(question)
        intersection = len(set(user_tokens).intersection(question_tokens))
        union = len(set(user_tokens).union(question_tokens))
        sim = intersection / union
        if sim > max_sim:
            max_sim = sim
            most_similar_question = question
    return most_similar_question

# Streamlit app
def main():
    st.title("BODYBALANCE.AI")
    st.write("Hello! I'm a chatbot designed by Clifford.")
    st.write ("How can i help you today ?")

    st.write ("You can also choose from the options below:")
    st.write ("About BodyBalance| Product Information | Product catalog | Ordering Process | Shipping and Delivery | Return Policy | Technical Support | Contact and Assistance | Special Offers and Promotions")

    input_mode = st.radio("Select Input Mode:", ("Text", "Speech"))

    if input_mode == "Text":
        user_input = st.text_input("User:")
        if st.button("Submit"):
            similar_question = find_similar_question(user_input)
            if similar_question:
                st.write("Chatbot:", qa_pairs[similar_question])
            else:
                st.write("Chatbot: Sorry, I couldn't find a relevant answer.")
    else:  # Speech input
        st.write("Speech input is not yet supported.")

if __name__ == "__main__":
    main()
