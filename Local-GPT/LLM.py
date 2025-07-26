import re
import os
import ollama
import random
from pydub.playback import play
from dotenv import load_dotenv
from Client import FilteringTTS, STT, localSTT
import speech_recognition as sr
from pydub import AudioSegment
from langchain_openai import ChatOpenAI
from langchain.document_loaders import CSVLoader, TextLoader, PyPDFLoader
from langchain.chains import LLMChain, ConversationChain
from langchain_ollama import OllamaLLM, ChatOllama
from langchain.chains import ConversationChain
from langchain.chains import create_retrieval_chain
from system_prompt import system_message
from langchain.vectorstores import Chroma, FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
import pandas as pd

## important initializations
load_dotenv()
# Get the API key from the .env file
Oapi_key = os.getenv("OPENAI_API_KEY")
embedding = OpenAIEmbeddings(api_key=Oapi_key, model="text-embedding-3-large")
embeddingADA = OpenAIEmbeddings(api_key=Oapi_key, model="text-embedding-ada-002")

# Past Conversations Loading (Dummy)
UserMessages = {
    1: [
        {"role": "ai", "content": "Hello Dear"},
        {"role": "human", "content": "Hi"},
    ],
    2: [] 
}

memories = {}
UserID = 1  # Default user ID for testing

#Main Prompt
prompt = ChatPromptTemplate([
    ("system", system_message),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),  
])

# loading past conv into the buffers
def InitializingMemories(userId, memtype):
    history = InMemoryChatMessageHistory()
    user_msgs = UserMessages.get(userId, [])

    for msg in user_msgs:
        if msg["role"] == "ai":
            history.add_ai_message(msg["content"])
        elif msg["role"] == "human":
            history.add_user_message(msg["content"])

    if memtype == 1:
        memory = ConversationBufferMemory(memory_key="history", return_messages=True, chat_memory=history)
    elif memtype == 2:
        memory = ConversationBufferWindowMemory(memory_key="history", return_messages=True, chat_memory=history, k=10)

    return memory


# RAG Pipeline
def RAG(csv):
    vectors = None  
    df = pd.read_csv(csv, usecols=[0, 1])
    df = df.dropna()
    df.columns = ['input', 'output']
    print(df.shape)  # Show number of rows and columns
    print(df.head(10))  # Show first 10 rows
    print(df.dtypes)  # Show column types

    batch_size = 10000
    num_rows = len(df)  
    num_batches = (num_rows + batch_size - 1) // batch_size  
    partition = [
        Document(page_content=row["input"], metadata={"response": row["output"]})
        for _, row in df.iterrows()
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    split_docs = text_splitter.split_documents(partition)

    for i in range(num_batches):
        StartIndex = i * batch_size
        end_idx = min((i + 1) * batch_size, num_rows)  
        batch_docs = split_docs[StartIndex:end_idx] 
        
        print(f"Embedding rows {StartIndex} to {end_idx}...")
        batch_vectors = FAISS.from_documents(batch_docs, embedding=embeddingADA)  
        if vectors is None:
            vectors = batch_vectors
        else:
            vectors.merge_from(batch_vectors)
        print(f"Batch {i+1} completed ✅ (Rows {StartIndex} to {end_idx})")
    vectors.save_local("AlignedResponsesFiltered_RagDocADA") # save the index to load it later
    exit(1)

def LoadVectors(path):
    vectors = FAISS.load_local(path, embeddings=embedding, allow_dangerous_deserialization=True) # loading the faiss index
    return vectors
    
def LLMS():
    #GPT-4o
    MainLLM = ChatOpenAI(model='gpt-4o', api_key=Oapi_key, temperature=0.6) # the higher the temperature, the more creative the response
    #LLAMA
    Llama3_3 = OllamaLLM(model="llama3.3")
    Llama3_2 = OllamaLLM(model="llama3.2")
    #Deepseek
    Deepseek = OllamaLLM(model="deepseek-r1:32b")
    # (Gemma3)
    gemma3 = OllamaLLM(model = "gemma3")

    return MainLLM,Llama3_3,Llama3_2,Deepseek,gemma3

# RAG Retrieval
def retrieve_response(prompt):
    ConcatenatedResponses = ""
    count = 0
    similar_docs = vectors.similarity_search(prompt, k=1)  # Get the most relevant match
    for doc in similar_docs:
        print(f"Retrieved Document: {doc.page_content} -> {doc.metadata['response']}")

    if similar_docs:
        for i in similar_docs:
            count=count+1
            ConcatenatedResponses = ConcatenatedResponses + str(count) + " " + i.metadata['response'] + "\n"
        return ConcatenatedResponses
    else:
        return "I'm here to help, but I don't have an answer for that yet."

# Main Chain
def GetChain(systemPrompt, llm, userId, memoryType):
    memory = InitializingMemories(userId=userId, memtype=memoryType)
    memories[userId] = memory 
    chat = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=systemPrompt,
        verbose=True,    
    )
    return chat, memory

def Run(userID, user_input, system_message, chat):

    retrievedText = retrieve_response(user_input)

    # all the exitting phrases
    if user_input in exitting_phrases:
        FilteringTTS("Goodbye! See you soon :)", "BotAudio.wav")
        sound = AudioSegment.from_file("BotAudio.wav")
        play(sound)        
        exit(1)

    updated_system_message = f"""
    {system_message} \n
    \n### Provided Similar Responses: \n{retrievedText} \n """ # update the prompt with the retrieved text

    updated_prompt = ChatPromptTemplate.from_messages([
        ("system", updated_system_message),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chat.prompt = updated_prompt
    response = chat.invoke({
    "input": f"{user_input}\n"  # embed retrieved docs in input
})
     
    clean_response = re.sub(r"<think>.*?</think>\s*", "", response['response'], flags=re.DOTALL) # **only for O1 & deepsek R1**

    print(clean_response)
    FilteringTTS(clean_response, "BotAudio.wav")
    sound = AudioSegment.from_file("BotAudio.wav")
    play(sound)
    return clean_response

exitting_phrases = ["goodbye", "Goodbye", "bye", "Bye", "exit", "Exit", "leave", "Leave", "stop", "Stop", "quit", "Quit"]

# RAG("/home/group02-f24/Documents/Khalil/Datasets/AllDAIC/aligned_responses_filtered.csv") # loading the RAG database
vectors = LoadVectors("/Users/muhammadabdelmohsen/Desktop/University/Spring 25/Thesis/LLM/RAG_Vectors/Index_Large_Faiss")

# All LLMS (Choose whatever you want in the llm parameter in run function)
MainLLM, Llama3_3, Llama3_2, Deepseek, gemma3 = LLMS()

chain, mem = GetChain(systemPrompt=prompt, llm= MainLLM, userId=UserID, memoryType=1) #now every user has his own chat chain

while True:
    counter=0
    choice = input("Enter 1 for Text or 2 for Voice: ")
    while(counter!=1):
        if choice == '1':
            text = input("Enter your Prompt >> ")
            Run(userID=UserID, user_input=text, system_message = system_message, chat=chain)
            counter=1
        elif choice == '2':     
            text = localSTT() #initializing the mic for the bot
            Run(userID=UserID, user_input=text, system_message = system_message, chat=chain)
            counter=1
        else:
            print("Invalid choice, Please Re-enter")
            choice = input("Enter 1 for Text or 2 for Voice: ")
