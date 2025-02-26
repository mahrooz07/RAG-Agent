<h1>Document Query Assistant with LangChain and Ollama</h1>
<p>This project demonstrates how to build a document query assistant using machine learning and natural language processing. The assistant is built using LangChain for document processing, vectorization, and similarity search, with Ollama models for generating responses. The assistant answers user queries based on the content of an uploaded PDF document. Here’s a step-by-step explanation of how I implemented this project:</p>


<h2>1. Importing Necessary Libraries</h2>

<pre>
import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PDFPlumberLoader
</pre>

<p>This section imports the necessary libraries for the project. The <code>streamlit</code> library is used for building the interactive user interface (UI). The LangChain libraries handle tasks like text processing, document chunking, embedding generation, and similarity search, allowing the model to interact with the uploaded PDF documents and answer user queries.</p>

<h2>2. Customizing the Streamlit UI</h2>

<pre>
st.markdown("""
    <style>
    .stApp { ... }
    .stChatInput input { ... }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) { ... }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) { ... }
    .stFileUploader { ... }
    h1, h2, h3 { ... }
    </style>
""", unsafe_allow_html=True)
</pre>

<p>The UI is customized with CSS styles to modify the appearance of the chat interface. This includes styling the background color, text colors, input fields, and chat messages for a visually appealing and user-friendly experience.</p>

<h2>3. Defining the Prompt Template</h2>

<pre>
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query : {user_query}
Context: {document_context}
Answer:
"""
</pre>

<p>This is the <strong>prompt template</strong> that guides the language model (LLM) in generating concise, factual answers to user queries. The context from the document is provided along with the user's question to ensure the LLM understands the context and responds appropriately.</p>

<h2>4. Saving the Uploaded File</h2>

<pre>
def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path,"wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path
</pre>

<p>This function is responsible for saving the uploaded PDF file to the server's local storage, so that it can be processed and used for answering queries.</p>

<h2>5. Loading the PDF Document</h2>

<pre>
def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()
</pre>

<p>This function uses <code>PDFPlumberLoader</code> to load and extract the content from the uploaded PDF file.</p>

<h2>6. Splitting the Document into Chunks</h2>

<pre>
def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )
    return text_processor.split_documents(raw_documents)
</pre>

<p>This function splits the raw PDF content into smaller, manageable chunks for easier processing. The chunk size is set to 1000 characters, with an overlap of 200 characters between chunks. This ensures the model has enough context when processing each chunk.</p>

<h2>7. Indexing the Documents</h2>

<pre>
def index_documents(document_chunks):
    return DOCUMENT_VECTOR_DB.add_documents(document_chunks)
</pre>

<p>This function adds the document chunks to an in-memory vector store, which allows the application to later perform similarity searches and retrieve relevant chunks based on the user's query.</p>

<h2>8. Finding Related Documents</h2>

<pre>
def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)
</pre>

<p>This function searches the vector store for the most relevant document chunks based on the similarity to the user's query. It returns the top matching chunks.</p>

<h2>9. Generating the Answer</h2>

<pre>
def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})
</pre>

<p>This function combines the user's query and the context from related document chunks to generate an answer using the language model. The model is prompted with the query and context to produce a relevant and concise response.</p>

<h2>10. Handling User Input and Displaying the Output</h2>

<pre>
uploaded_pdf = st.file_uploader(
    "Upload your Document(PDF)",
    type = "pdf",
    help = "Upload a pdf file.",
    accept_multiple_files = False
)
</pre>

<p>This section of the code sets up the user interface for uploading a PDF document. The file uploader accepts only PDF files, and the user can upload one file at a time.</p>

<h2>11. Processing the Document and Generating Responses</h2>

<pre>
if uploaded_pdf:
    file_path = save_uploaded_file(uploaded_pdf)
    raw_documents = load_pdf_documents(file_path)
    document_chunks = chunk_documents(raw_documents)
    index_documents(document_chunks)

    st.success("✅ Document processed successfully! You can ask your questions.")
    user_input = st.chat_input("Ask a question about your document")
</pre>

<p>Once the PDF is uploaded, the document is processed in the following sequence: saved, loaded, chunked, and indexed. After processing, the user is informed that the document is ready, and they can begin asking questions related to the document.</p>

<h2>12. Generating the Assistant's Response</h2>

<pre>
if user_input:
    with st.chat_message("user"):
        st.write(user_input)  

    with st.spinner("Generating answer..."):
        related_docs = find_related_documents(user_input)
        response = generate_answer(user_input, related_docs)

    with st.chat_message("assistant", avatar="🤖"):
        st.write(response)
</pre>

<p>Once the user inputs a query, the system retrieves related documents using a similarity search and then generates an answer based on those documents. The result is then displayed as a response from the assistant.</p>

<h2>13. Summary</h2>

<p>This project successfully combines Streamlit and LangChain to build a powerful research assistant capable of answering questions based on the content of uploaded PDF documents. The system handles document processing, chunking, indexing, similarity searching, and generating responses based on user queries. The project showcases how to integrate machine learning and natural language processing with an interactive UI to create a seamless user experience.</p>
