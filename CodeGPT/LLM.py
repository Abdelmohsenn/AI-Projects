import random, json, re, os
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain_ollama import OllamaLLM, ChatOllama
from langchain.chains import ConversationChain
from langchain.vectorstores import FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
import pandas as pd

class Client:

    def __init__(self, apikey, system_message, llmName, embedderName):
        ## important initializations
        self.APIKEY = apikey
        self.llmName = llmName
        self.embedderName = embedderName
        self.embedder = None
        self.client = None
        self.sysMessage = system_message

    def setPrompt(self):
        self.prompt = ChatPromptTemplate([

            ("system", self.sysMessage),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])    
        
    def __repr__(self):
        return f"Client Powered by {self.llmName} LLM and {self.embedderName} Embedding Model"

    def setClient(self, temperature):
        self.client = ChatOpenAI(model = self.llmName, api_key=self.APIKEY, temperature=temperature)        

    def EmbeddingClient(self):
        self.embedder = OpenAIEmbeddings(api_key=self.APIKEY, model=self.embedderName)

    # loading past conv into the buffers
    def initializeMemory(self, memType):
        UserMessages = []
        history = InMemoryChatMessageHistory()
        with open("CodeGPT/History.json", "r") as file:
            json_data = file.read()
            UserMessages.append(json.loads(json_data))
            index = self.user_id - 1  # Adjust index for zero-based indexing
            user_conversations = UserMessages[0][index].get('history', [])  # List of conversation blocks
            print("User Conversations Loaded:", user_conversations)

        for msg in user_conversations:
            if msg["role"] == "ai":
                history.add_ai_message(msg["content"])
            elif msg["role"] == "human":
                history.add_user_message(msg["content"])

        if memType == 1:
            self.memory = ConversationBufferMemory(memory_key="history", return_messages=True, chat_memory=history)
        elif memType == 2:
            self.memory = ConversationBufferWindowMemory(memory_key="history", return_messages=True, chat_memory=history, k=10)

    def setUser(self, user_id, memType):
       self.user_id = user_id
       self.initializeMemory(memType)

    # RAG Pipeline Function
    def RAG(self, csv):
        self.vectors = None  
        df = pd.read_csv(csv, usecols=[0, 1])
        df = df.dropna()
        df.columns = ['input', 'output']

        batch_size = 10000 # set batch size
        num_rows = len(df)  
        num_batches = (num_rows + batch_size - 1) // batch_size  
        partition = [
            Document(page_content=row["input"], metadata={"response": row["output"]}) # embed a metadata for each input
            for _, row in df.iterrows()
        ]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        split_docs = text_splitter.split_documents(partition)

        for i in range(num_batches):
            StartIndex = i * batch_size
            end_idx = min((i + 1) * batch_size, num_rows)  
            batch_docs = split_docs[StartIndex:end_idx] 
            
            print(f"Embedding rows {StartIndex} to {end_idx}...")
            batch_vectors = FAISS.from_documents(batch_docs, embedding=self.embedder)  
            if self.vectors is None:
                self.vectors = batch_vectors
            else:
                self.vectors.merge_from(batch_vectors)
            print(f"Batch {i+1} completed ✅ (Rows {StartIndex} to {end_idx})")
        self.vectors.save_local("AlignedResponsesFiltered_RagDocADA") # save the index to load it later
        exit(1)

    # loading the available vectorStore
    def LoadVectors(self, path):
        if self.embedder is None:
            self.EmbeddingClient()
        self.vectors = FAISS.load_local(path, embeddings=self.embedder, allow_dangerous_deserialization=True) # loading the faiss index
        return self.vectors

    # RAG Retrieval
    def retrieve_response(self, prompt, vecStore):
        ConcatenatedResponses = ""
        count = 0
        similar_docs = self.vectors.similarity_search(prompt, k=1)  # Get the most relevant match
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
    def GetChain(self, temperature, user_id, memory_type, path):
        self.LoadVectors(path)
        self.setClient(temperature=temperature)
        self.setUser(user_id, memory_type)
        self.setPrompt()
        chain = ConversationChain(
            llm=self.client,
            memory=self.memory,
            prompt=self.prompt,
            verbose=True,    
        )
        return chain
