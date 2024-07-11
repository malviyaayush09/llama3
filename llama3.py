##(only pdf)

# import streamlit as st
# from langchain_community.document_loaders import UnstructuredPDFLoader
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.chat_models import ChatOllama
# from langchain_core.runnables import RunnablePassthrough
# from langchain.retrievers.multi_query import MultiQueryRetriever

# # Streamlit UI
# st.title("PDF Query with LangChain")

# uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# if uploaded_file is not None:
#     # Save the uploaded file temporarily
#     with open("temp.pdf", "wb") as f:
#         f.write(uploaded_file.read())
    
#     # Load and process the PDF
#     loader = UnstructuredPDFLoader(file_path="temp.pdf")
#     data = loader.load()
    
#     # Split and chunk the text
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
#     chunks = text_splitter.split_documents(data)
    
#     # Add to vector database
#     vector_db = Chroma.from_documents(
#         documents=chunks,
#         embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
#         collection_name="local-rag"
#     )
    
#     # Set up the LLM and retriever
#     local_model = "llama3"
#     llm = ChatOllama(model=local_model)
    
#     QUERY_PROMPT = PromptTemplate(
#         input_variables=["question"],
#         template="""You are an AI language model assistant. Your task is to generate five
#         different versions of the given user question to retrieve relevant documents from
#         a vector database. By generating multiple perspectives on the user question, your
#         goal is to help the user overcome some of the limitations of the distance-based
#         similarity search. Provide these alternative questions separated by newlines.
#         Original question: {question}"""
#     )
    
#     retriever = MultiQueryRetriever.from_llm(
#         vector_db.as_retriever(),
#         llm,
#         prompt=QUERY_PROMPT
#     )
    
#     # Define the RAG prompt
#     template = """Answer the question based ONLY on the following context:
#     {context}
#     Question: {question}
#     """
    
#     prompt = ChatPromptTemplate.from_template(template)
    
#     chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
    
#     question = st.text_input("Enter your question:")
    
#     if question:
#         response = chain.invoke(question)
#         st.write(response)



##(pdf and normal prompting with history for chat not pdf)
# import streamlit as st
# from langchain_community.document_loaders import UnstructuredPDFLoader
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.chat_models import ChatOllama
# from langchain_core.runnables import RunnablePassthrough
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain_community.llms import Ollama
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# # Streamlit UI
# st.title("LangChain App")

# # Sidebar for navigation
# option = st.sidebar.selectbox(
#     'Select an option',
#     ('PDF Query', 'Chatbot')
# )

# if option == 'PDF Query':
#     st.header("PDF Query")

#     uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

#     if uploaded_file is not None:
#         # Save the uploaded file temporarily
#         with open("temp.pdf", "wb") as f:
#             f.write(uploaded_file.read())
        
#         # Load and process the PDF
#         loader = UnstructuredPDFLoader(file_path="temp.pdf")
#         data = loader.load()
        
#         # Split and chunk the text
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
#         chunks = text_splitter.split_documents(data)
        
#         # Add to vector database
#         vector_db = Chroma.from_documents(
#             documents=chunks,
#             embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
#             collection_name="local-rag"
#         )
        
#         # Set up the LLM and retriever
#         local_model = "llama3"
#         llm = ChatOllama(model=local_model)
        
#         QUERY_PROMPT = PromptTemplate(
#             input_variables=["question"],
#             template="""You are an AI language model assistant. Your task is to generate five
#             different versions of the given user question to retrieve relevant documents from
#             a vector database. By generating multiple perspectives on the user question, your
#             goal is to help the user overcome some of the limitations of the distance-based
#             similarity search. Provide these alternative questions separated by newlines.
#             Original question: {question}"""
#         )
        
#         retriever = MultiQueryRetriever.from_llm(
#             vector_db.as_retriever(),
#             llm,
#             prompt=QUERY_PROMPT
#         )
        
#         # Define the RAG prompt
#         template = """Answer the question based ONLY on the following context:
#         {context}
#         Question: {question}
#         """
        
#         prompt = ChatPromptTemplate.from_template(template)
        
#         chain = (
#             {"context": retriever, "question": RunnablePassthrough()}
#             | prompt
#             | llm
#             | StrOutputParser()
#         )
        
#         question = st.text_input("Enter your question:")
        
#         if question:
#             response = chain.invoke(question)
#             st.write(response)

# elif option == 'Chatbot':
#     st.header("Chatbot")

#     llm = Ollama(model="llama3")

#     chat_history = []

#     prompt_template = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 "You are an AI named Mike, you answer questions with simple answers and no funny stuff.",
#             ),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{input}"),
#         ]
#     )

#     chain = prompt_template | llm

#     if 'chat_history' not in st.session_state:
#         st.session_state['chat_history'] = []

#     def get_response(question):
#         response = chain.invoke({"input": question, "chat_history": st.session_state['chat_history']})
#         st.session_state['chat_history'].append(HumanMessage(content=question))
#         st.session_state['chat_history'].append(AIMessage(content=response))
#         return response

#     question = st.text_input("You: ")

#     if st.button("Send"):
#         if question:
#             response = get_response(question)
#             st.write("AI: " + response)


import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder

# Streamlit UI
st.title("PDF Query with LangChain")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    # Load and process the PDF
    loader = UnstructuredPDFLoader(file_path="temp.pdf")
    data = loader.load()
    
    # Split and chunk the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    
    # Add to vector database
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        collection_name="local-rag"
    )
    
    # Set up the LLM and retriever
    local_model = "llama3"
    llm = ChatOllama(model=local_model)
    
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}"""
    )
    
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )
    
    # Define the RAG prompt
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    chat_history = []

    # Chat history prompt template
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI named Mike, you answer questions with simple answers and no funny stuff.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    question = st.text_input("Enter your question:")
    
    if question:
        response = chain.invoke({"input": question, "chat_history": chat_history})
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response))

        st.write(response)
