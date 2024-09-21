import os
import re
import time

import streamlit as st
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Loading HuggingFace and OpenAI:
load_dotenv(".env")
OPENAIKEY = os.getenv("OPENAI_API_KEY")

st.title("Flashcard Generator")

def generate_flashcards(user_text):
    # Step 3: Use the text to generate flashcards using OpenAI's GPT-3 API
    model = ChatOpenAI(temperature=1, openai_api_key=OPENAIKEY)

    # System message providing instruction format for the AI
    chat_history = [
        SystemMessage('''Generate 1 flashcard with both the question and answer being less than two sentences. Use the format provided below:

                        Flashcard 1: What are the three phases of interphase?
                        Answer: Gap 1 (G1), DNA synthesis (S), and Gap 2 (G2).
                        Flashcard 2: What happens during interphase?
                        Answer: Preparations for cell division occur, including growth, duplication of most cellular contents, and replication of DNA.
                        Flashcard 3: What is mitosis?
                        Answer: Mitosis is the process of active division in which duplicated chromosomes attach to spindle fibers, align along the equator of the cell, and then separate.'''),
        HumanMessage(user_text)
    ]

    retries = 5
    for attempt in range(retries):
        try:
            flashcards = model(chat_history).content
            break  # Exit loop if successful
        except Exception as e:
            if e.code == 'insufficient_quota':
                if attempt < retries - 1:  # If this isn't the last attempt
                    wait_time = 2 ** attempt  # Exponential backoff
                    st.warning(f"Rate limit reached. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    st.error("Exceeded quota. Please try again later or consider reducing the number of flashcards requested.")
                    return []
            else:
                st.error("An error occurred: " + str(e))
                return []

    # Use regex to extract flashcards based on the expected format
    flashcard_texts = re.findall(r'Flashcard \d+: (.*?)Answer: (.*?)(?:Flashcard|\Z)', flashcards, re.DOTALL)

    # Format flashcards into a clean list
    flashcards_list = [f"Flashcard {idx + 1}: {q.strip()} Answer: {a.strip()}" for idx, (q, a) in enumerate(flashcard_texts)]

    return flashcards_list

# Input box for user to provide text to generate flashcards from
user_text = st.text_input("Enter the text you want to generate flashcards for:")

if st.button("Generate Flashcards"):
    if user_text:
        flashcards_output = generate_flashcards(user_text)
        sorted_flashcards = sorted(flashcards_output)  # Sort the flashcards

        for flashcard in sorted_flashcards:
            st.write(flashcard)
    else:
        st.warning("Please enter some text to generate flashcards.")

# Streamlit 
# Step 4: Save the flashcards to a file (not yet implemented)
# Step 5: Use the HuggingFace API to upload the flashcards to the HuggingFace Hub (not yet implemented)
# Step 6: Share the link to the flashcards with the user (not yet implemented)
# Step 7: Allow the user to download the flashcards (not yet implemented)
