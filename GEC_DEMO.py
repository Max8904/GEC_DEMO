from train import Trainer
import streamlit as st

if __name__ == "__main__":
    trainer = Trainer(device='cuda', tags_file="all_tags.json",
                          filter_indices_file="indices.json")
    trainer.load_weights("epoch_4.ckpt")

    st.title("GEC Demo")
    option = st.selectbox("Choose a sentence or...",["","who are she"],help="Choose a sentence, any sentence.")
    # "where are he from?","how to do"

    sentence = st.text_input("type a sentence!", option, help="Type a sentence, any sentence.")
    sentence_corrected = trainer.interactive_eval(sentence)
    sentence_corrected = sentence_corrected.replace("[CLS] ", "")
    sentence_corrected = sentence_corrected.replace(" [SEP]", "")
    
    print(sentence)
    st.subheader(f"Original sentence: \n{sentence}")
    
    print(sentence_corrected)
    if(sentence_corrected!="[SEP]"):
        st.subheader(f"Corrected sentence: \n{sentence_corrected}")
        st.success('sentence has been corrected!')