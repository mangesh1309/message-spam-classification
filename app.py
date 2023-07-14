import streamlit as st
import nltk
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sklearn

nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

model = pickle.load(open("model.pkl", "rb"))
vectoriser = pickle.load(open("vectorizer.pkl", "rb"))  # tfidf


def text_transformer(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


st.title("Message Classifier")
input_msg = st.text_area("Enter Your Message Here")

if st.button("Predict"):
    # 1. Preprocess input message
    transformed_msg = text_transformer(input_msg)

    # 2. Vectorise the preprocessed message
    vectorised_msg = vectoriser.transform([transformed_msg])

    # 3. Predict using model
    result = model.predict(vectorised_msg)[0]
    # st.header(result)
    # 4. Display predicted results
    if result == 1:
        st.header("Spam!")
    else:
        st.header("Not Spam")


hide = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide, unsafe_allow_html=True)
