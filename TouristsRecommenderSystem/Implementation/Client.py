import pickle
# from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()
apikey = os.getenv("OPENAI_API_KEY") # load the OpenAI API key from the .env file

embeddingModel = OpenAIEmbeddings(api_key=apikey, model="text-embedding-3-small") # the emdding model using OpenAI's API
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # the text splitter to split documents into smaller chunks
# the system prompt
systemPrompt = """
You are a helpful assistant. Provide recommendations for the user based on their preferences.
User ID: {user_id}
Below is a list of recommended places based on the user's past interactions:

{recommendations}

For each place, give a brief description and list them one by one, clearly and concisely.

You have access to more detailed information, such as climate, city locations, and operating hours, for each place:
{Rec}
**Only** provide this extra information when the user explicitly asks for details about locations, features, climate, or operating hours. If not asked, do not mention or reveal this information.
"""

client = ChatOpenAI(model="gpt-4o", api_key=apikey, temperature=0.7, max_tokens=450) # the chat client using OpenAI's GPT-4o model
def RaggedVectorStore():
    columns_to_join = ['place_name', 'place_type', 'city', 'Operating_Hours']  # change these to your columns

    df = pd.read_csv("Dell/Implementation/AfterMerging/Ragged.csv")

    # Create documents by concatenating selected columns
    documents = [ # Create a list of Document objects
        Document(
            page_content=' '.join([str(row[col]) for col in columns_to_join]),
            metadata=row.to_dict()
        )
        for _, row in df.iterrows()
    ]

    # Create and save the FAISS vectorstore
    VectorStore = FAISS.from_documents(documents, embeddingModel)
    VectorStore.save_local("Dell/Implementation/PlacesVectorStore")
    return VectorStore

if not os.path.exists("Dell/Implementation/PlacesVectorStore"):
    VectorStore = None
else:
    VectorStore = RaggedVectorStore()

def model(): # Load the KNN model for recommendations
    with open('Dell/Implementation/Recommender_KNN.pkl', 'rb') as file: # replace it with pkl file path if needed (KNN) is more efficient for this task
        model = pickle.load(file)
    return model
def retrieve_places(query, vectorstore):    
    # Retrieve relevant documents from the vectorstore
    retrieved_docs = vectorstore.similarity_search(query, k=2)
    
    # Extract place names and descriptions from the retrieved documents
    results = []
    for doc in retrieved_docs:
        place_name = doc.metadata.get('place_name', 'Unknown Place')
        climate = doc.metadata.get('Climate', 'Unknown Climate')
        operating_hours = doc.metadata.get('Operating_Hours', 'Unknown Hours')
        city = doc.metadata.get('city', 'Unknown City')
        results.append(
            f"place_name: {place_name}, climate: {climate}, operating_hours: {operating_hours}, city: {city}"
        )
    
    return results

def datasets(): # initialize the datasets for users, places, and interactions

    # --------- USERS ---------
    users_df = pd.read_csv("Dell/Implementation/AfterMerging/users.csv")
    # --------- PLACES ---------
    places_df = pd.read_csv("Dell/Implementation/AfterMerging/places.csv")
    # --------- INTERACTIONS ---------
    interactions_df = pd.read_csv("Dell/Implementation/AfterMerging/interactions.csv")

    return users_df, places_df, interactions_df

PlacesToVisit = [] # List to store recommended places for the user
def recommend_for_user(model, interactions_df, user_id, top_n, places_df):
    all_place_ids = set(interactions_df['place_id'].unique())
    visited = set(interactions_df[interactions_df['user_id'] == user_id]['place_id'])
    unvisited = all_place_ids - visited

    predictions = []
    for place_id in unvisited:
        pred = model.predict(uid=user_id, iid=place_id)
        predictions.append((place_id, pred.est))
        
    # Sort
    top_recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    
    # Show names
    results = []
    for place_id, score in top_recommendations:
        place_name = places_df[places_df['place_id']==place_id]['place_name'].iloc[0]
        results.append({'place_id': place_id, 'place_name': place_name, 'predicted_rating': score})
    return results

user_id = 154# just an example user_id, you can change it to any valid user_id from your dataset
# Example usage
users_df, places_df, interactions_df = datasets()
# print(interactions_df['user_id'].unique())
recommendations = recommend_for_user(model(), interactions_df, user_id, top_n=1, places_df=places_df)
for i in recommendations:
    PlacesToVisit.append(i['place_name'])

input = "Give me a place to visit in Egypt"

# Example usage of the recommender system with RAG pipeline
retrievedInfo =  retrieve_places(input, VectorStore)
prompt = ChatPromptTemplate.from_messages([
        ("system", systemPrompt.format(user_id=user_id, recommendations=PlacesToVisit, Rec = str(retrievedInfo))),
        ("human", "{input}"),
    ])

chain = prompt | client

response = chain.invoke({"input": input})
print(f"Retrieved Information: {retrievedInfo}")
print(f"Recommendations for User {user_id}: {PlacesToVisit}")

print(f"The Final Response: {response.content}")

