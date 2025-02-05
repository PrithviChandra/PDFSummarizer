import streamlit as st
from langchain_openai import OpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

#Load API key
OpenAI.api_key = st.secrets["OPEN_API_KEY"]

#Summarize function
def summarize_pdf(pdf, chunk_size, chunk_overlap, prompt):
    #Invoking LLM model
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0, openai_api_key=OpenAI.api_key)

    #Loading PDF file to text extractor
    loader = PDFPlumberLoader(pdf)
    raw_text = loader.load()

    #Extract text
    raw_doc = [text.page_content for text in raw_text]

    #Splitting text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.create_documents(raw_doc)

    #Summarize each chunk
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
    summary = chain.invoke(chunks, return_only_outputs=True)

    return summary['output_text']

def main():
    st.set_page_config(page_title="Summarize PDF", page_icon=":book:", layout="wide")
    st.title("Summarize PDF")

    #Upload file
    uploaded_file = st.file_uploader("Upload the PDF file", type=["pdf"])

    if uploaded_file:
        st.write("Loaded PDF file successfully")

        #Inputting prompt
        user_prompt = st.text_input("Enter summary instructions:")
        prompt_complete = user_prompt + """ {text}"""
        prompt = PromptTemplate(input_variables=["text"], template=prompt_complete)

        if st.button("Generate Summary"):
            summary = summarize_pdf(uploaded_file, 1000, 20, prompt)
            st.write(summary)

if __name__ == "__main__":
    main()
