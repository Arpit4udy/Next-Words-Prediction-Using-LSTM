import streamlit as st
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import load_model
#loadd tokenizer and model
with open('tokenizer.pkl','rb') as f:
    tokenizer=pickle.load(f)
model=load_model('model.h5')    


max_len=28

reverse_index={idx:word for word,idx in tokenizer.word_index.items()}

def generate_text(seed_text,num_words=10):
    text=seed_text
    for _ in range(num_words):
        seq=tokenizer.texts_to_sequences([text])[0]
        padded=pad_sequences([seq],maxlen=max_len,padding='pre')
        preds=model.predict(padded,verbose=0)
        pos=np.argmax(preds)
        next_word=reverse_index.get(pos," ")
        text+=" "+next_word
    return text    

st.title("Next Word Prediction")
seed=st.text_input("Enter a Startinf text: ",'Hello')
num_words=st.slider("Number of Words",1,20,10,1)
if st.button("Generate"):
    result=generate_text(seed,num_words)
    st.write(result)
