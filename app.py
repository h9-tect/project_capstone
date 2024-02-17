import streamlit as st
from transformers import pipeline

# Initialize the translation and question-answering pipelines
translator = pipeline("translation_en_to_ar", model="Helsinki-NLP/opus-mt-en-ar")
qa_pipeline = pipeline("question-answering", model="timpal0l/mdeberta-v3-base-squad2")

def translate_and_answer(english_text, arabic_question):
    # Translate the text
    translated_text = translator(english_text, max_length=512)[0]['translation_text']
    
    # Perform question answering
    answer = qa_pipeline({
        'context': translated_text,
        'question': arabic_question
    })
    return translated_text, answer['answer']

# Streamlit UI
st.title('English to Arabic Translation and Question Answering')

english_text = st.text_area("Enter English text:", height=150)
arabic_question = st.text_input("Enter your question in Arabic:")

if st.button('Translate and Answer'):
    if english_text and arabic_question:
        translated_text, answer = translate_and_answer(english_text, arabic_question)
        st.write("Translated Text:")
        st.write(translated_text)
        st.write("Answer:")
        st.write(answer)
    else:
        st.write("Please enter both the text and the question.")
