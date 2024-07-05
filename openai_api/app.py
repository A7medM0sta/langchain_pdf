from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import openai
from retrying import retry



class RetryDecorator:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def _retry_if_openai_timeout_exception(self, exception):
        """Check if we should retry on openai.Timeout exception."""
        return isinstance(exception, openai.OpenAIError) and 'timeout' in str(exception).lower()

    def create(self):
        """Create a retry decorator."""
        return retry(
            retry_on_exception=self._retry_if_openai_timeout_exception,
            stop_max_attempt_number=3,
            wait_fixed=2000
        )

class BaseModel:
    def get_embeddings(self, text):
        """Generate embeddings for the given text."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def answer_question(self, question, context):
        """Answer a question based on the given context."""
        raise NotImplementedError("This method should be implemented by subclasses.")

class OpenAIModel(BaseModel):
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()  # Assuming this is a class that handles OpenAI embeddings

    def get_embeddings(self, text):
        return self.embeddings.embed_documents([text])[0]

    def answer_question(self, question, context):
        # Assuming `load_qa_chain` and `get_openai_callback` are functions that handle question answering
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=[context], question=question)
        return response

from transformers import pipeline

class HuggingFaceModel(BaseModel):
    def __init__(self):
        self.embedder = pipeline("feature-extraction", model="bert-base-uncased")
        self.question_answerer = pipeline("question-answering")

    def get_embeddings(self, text):
        return self.embedder(text)[0]

    def answer_question(self, question, context):
        return self.question_answerer(question=question, context=context)['answer']
class PDFQuestionAsker:
    def __init__(self):
        load_dotenv()
        st.set_page_config(page_title="Ask your PDF")
        self.model = None

    def select_model(self):
        """Let the user select the model from the sidebar."""
        model_type = st.sidebar.selectbox("Select the model", ["OpenAI", "Hugging Face"])
        if model_type == "OpenAI":
            self.model = OpenAIModel()
        elif model_type == "Hugging Face":
            self.model = HuggingFaceModel()

    def run(self):
        """Run the application with model selection."""
        self.select_model()
        text = self.upload_and_process_pdf()
        if text:
            chunks = self.split_text_into_chunks(text)
            # Assuming `create_embeddings_and_knowledge_base` uses `self.model.get_embeddings`
            knowledge_base = self.create_embeddings_and_knowledge_base(chunks)
            self.ask_question(knowledge_base)

if __name__ == '__main__':
    asker = PDFQuestionAsker()
    asker.run()