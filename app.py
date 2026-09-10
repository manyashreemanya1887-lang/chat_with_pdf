import io
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


load_dotenv()

st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")


@st.cache_resource
def create_embeddings():
	return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def read_pdfs(pdf_files):
	documents = []

	for pdf_file in pdf_files:
		reader = PdfReader(io.BytesIO(pdf_file.getvalue()))
		for page_number, page in enumerate(reader.pages, start=1):
			text = page.extract_text() or ""
			if text.strip():
				documents.append(
					Document(
						page_content=text,
						metadata={"source": pdf_file.name, "page": page_number},
					)
				)

	return documents


def build_vector_store(documents):
	splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
	chunks = splitter.split_documents(documents)
	if not chunks:
		raise ValueError("No readable text was found in the uploaded PDF files.")
	return FAISS.from_documents(chunks, create_embeddings())


def get_api_key():
	return os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")


def answer_question(question, vector_store):
	api_key = get_api_key()
	if not api_key:
		raise RuntimeError(
			"GOOGLE_API_KEY is not set. Add it to a .env file or configure it in "
			"Streamlit secrets."
		)

	documents = vector_store.similarity_search(question, k=10)
	context = "\n\n".join(document.page_content for document in documents)
	prompt = PromptTemplate(
		template="""
You are an AI assistant.
Answer the question using only the context below.
If the answer is not present in the context, say exactly:
THE ANSWER IS NOT AVAILABLE IN THE PROVIDED CONTEXT.
Answer in bullet points. Each point must be no longer than 200 words.
Explain the answer so a class 10 student can understand it.

Context:
{context}

Question:
{question}

Answer:
""",
		input_variables=["context", "question"],
	)
	llm = ChatGoogleGenerativeAI(
		model="gemini-3.1-flash-lite",
		temperature=1.9,
		api_key=api_key,
	)
	response = llm.invoke(prompt.format(context=context, question=question))
	return response.content, documents


def main():
	st.title("Chat with your PDFs")
	st.write("Upload one or more PDFs, then ask questions grounded in their content.")

	uploaded_files = st.file_uploader(
		"Choose PDF files",
		type="pdf",
		accept_multiple_files=True,
	)

	if uploaded_files and st.button("Process PDFs", type="primary"):
		with st.spinner("Reading and indexing your PDFs..."):
			try:
				documents = read_pdfs(uploaded_files)
				st.session_state.vector_store = build_vector_store(documents)
				st.session_state.document_names = [file.name for file in uploaded_files]
				st.success(f"Indexed {len(uploaded_files)} PDF file(s).")
			except Exception as error:
				st.error(f"Could not process the PDFs: {error}")

	if st.session_state.get("document_names"):
		st.caption("Indexed: " + ", ".join(st.session_state.document_names))

	question = st.text_input("Ask a question")
	if question and st.session_state.get("vector_store"):
		with st.spinner("Searching the PDFs and generating an answer..."):
			try:
				answer, sources = answer_question(
					question, st.session_state.vector_store
				)
				st.markdown(answer)
				with st.expander("Sources"):
					for source in sources:
						metadata = source.metadata
						st.write(
							f"{metadata.get('source', 'Unknown')} "
							f"(page {metadata.get('page', '?')})"
						)
			except Exception as error:
				st.error(str(error))
	elif question:
		st.info("Process at least one PDF before asking a question.")


if __name__ == "__main__":
	main()
