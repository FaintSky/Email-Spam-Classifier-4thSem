import streamlit as st
import pickle
import string
import nltk
import email
import xml.etree.ElementTree as ET
from email import policy
from email.parser import BytesParser
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    text = [ps.stem(i) for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    return " ".join(text)

def extract_text_from_eml(eml_file):
    msg = BytesParser(policy=policy.default).parse(eml_file)
    if msg.is_multipart():
        return ''.join(part.get_payload(decode=True).decode(errors='ignore') for part in msg.walk() if part.get_content_type() == "text/plain")
    else:
        return msg.get_payload(decode=True).decode(errors='ignore')

def extract_text_from_xml(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        return " ".join(root.itertext())  # Extract all text from XML
    except ET.ParseError:
        return ""

st.title("Email/SMS Spam Classifier")

# User input section
option = st.radio("Choose input type:", ("Manual Text", "Upload File (.txt/.eml/.xml)"))

input_sms = ""  # Initialize input_sms

if option == "Manual Text":
    input_sms = st.text_area("Enter the message")
elif option == "Upload File (.txt/.eml/.xml)":
    uploaded_file = st.file_uploader("Upload a .txt, .eml, or .xml file", type=["txt", "eml", "xml"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".txt"):
            input_sms = uploaded_file.read().decode("utf-8")
        elif uploaded_file.name.endswith(".eml"):
            input_sms = extract_text_from_eml(uploaded_file)
        elif uploaded_file.name.endswith(".xml"):
            input_sms = extract_text_from_xml(uploaded_file)

# Show the Predict button for all input types
if st.button("Predict") and input_sms.strip():  # Ensure input is not empty
    # Preprocessing
    transformed_sms = transform_text(input_sms)

    # Vectorizing
    vector_input = tfidf.transform([transformed_sms])

    # Prediction
    result = model.predict(vector_input)[0]

    # Display
    st.header("Spam." if result == 1 else "Not spam.")
