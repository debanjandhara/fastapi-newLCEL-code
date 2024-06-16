import os
import openai
import bs4
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from db import *

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Logging Function
from datetime import datetime

def log_error(error_id, function, reason):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("errorfile.txt", "a") as f:
        f.write(f"Timestamp: {timestamp}, Error ID: {error_id}, Function: {function}, Reason: {reason}\n")

# Initialize the model and retriever
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
print("LLM initialized")

# Construct retriever
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
print("Loading documents")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
print("Documents loaded and indexed")

# Contextualize question prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
print("History aware retriever created")

# Answer question prompt
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
print("RAG chain created")

# Statefully manage chat history
store = {}

# Keeping the session_id as History_id or else OpenAI couldn't understand
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # store = {}
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        recent_messages = get_recent_messages(session_id)
        for sender, message_text in reversed(recent_messages):
            if sender == 'User':
                store[session_id].add_user_message(message_text)
            elif sender == 'AI':
                store[session_id].add_ai_message(message_text)
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
print("Conversational RAG chain created")

# Invocation function
def invoke_rag_chain(input_text, history_id=None):
    store = {}
    try:
        if history_id:
            response = conversational_rag_chain.invoke(
                {"input": input_text, "chat_history": get_recent_messages(history_id)},
                config={
                    "configurable": {"session_id": history_id}
                },
            )
            print("\n\nget_recent_messages(history_id) Var --> ", get_recent_messages(history_id))
        else:
            response = conversational_rag_chain.invoke( 
                {"input": input_text},
                config={
                    "configurable": {"session_id": history_id}
                },
            )
        answer = response["answer"]
        
        print("\n\nStore Var --> ", store)
        
        # # Save user message and AI response to database
        insert_message(history_id, 'User', input_text)
        insert_message(history_id, 'AI', answer)

        export_messages_to_csv()
        export_messages_to_csv2()
        
        return answer
    except Exception as e:
        error_id = "ERR009"
        log_error(error_id, "invoke_rag_chain", str(e))
        print(f"Error {error_id} occurred during RAG chain invocation: {str(e)}")
        return None
