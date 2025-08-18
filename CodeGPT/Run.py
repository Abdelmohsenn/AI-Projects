from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from VoiceAssistant import VoiceAssistant
from pydub import AudioSegment
from LLM import Client
from system_prompt import system_message
from dotenv import load_dotenv
from pydub.playback import play
import os

load_dotenv() # load all the environment variables
api_key = os.getenv("OPENAI_API_KEY")
voiceAssistant = VoiceAssistant() # Initialize Voice Assistant
vectorStore = "/Users/muhammadabdelmohsen/Desktop/University/Spring 25/Thesis/LLM/RAG_Vectors/Index_Large_Faiss"
exitting_phrases = ["goodbye", "Goodbye", "bye", "Bye", "exit", "Exit", "leave", "Leave", "stop", "Stop", "quit", "Quit"]
client = Client(apikey=api_key, system_message=system_message, llmName='gpt-4o', embedderName='text-embedding-3-large') # initialize client
print(client)
chain = client.GetChain(temperature=0.7, user_id=1, memory_type=1, path=vectorStore)


def Run(user_input, system_message, chat, VecStore):

    retrievedText = client.retrieve_response(user_input, VecStore)

    # all the exitting phrases
    if user_input in exitting_phrases:
        voiceAssistant.TTS("Goodbye! See you soon for another journey of coding! :)", "BotAudio.wav")
        sound = AudioSegment.from_file("BotAudio.wav")
        play(sound)        
        exit(1)

    updated_system_message = f"""{system_message} \n\n ### Similar Retrieved Topic: \n {retrievedText} \n """ # update the prompt with the retrieved text
    chat.prompt = ChatPromptTemplate.from_messages([("system", updated_system_message), MessagesPlaceholder(variable_name="history"),("human", "{input}")]) # update the chat prompt
    response = chat.invoke({"input": f"{user_input}\n"})['response']  # embed retrieved docs in input
    voiceAssistant.TTS(response, "BotAudio.wav")  # Convert response to speech
    sound = AudioSegment.from_file("BotAudio.wav")
    return response, sound

while True:
    counter=0
    choice = input("Enter 1 for Text or 2 for Voice: ")
    while(counter!=1):
        if choice == '1':
            text = input("Enter your Prompt >> ")
            Response, sound = Run(user_input=text, system_message = system_message, chat=chain, VecStore=vectorStore)
            counter=1
        elif choice == '2':     
            text = voiceAssistant.localSTT()
            Response, sound = Run(user_input=text, system_message = system_message, chat=chain, VecStore=vectorStore)
            counter=1
        else:
            print("Invalid choice, Please Re-enter")
            choice = input("Enter 1 for Text or 2 for Voice: ")
    print(f"CodeGPT: {Response}")
    play(sound)  # Play the sound response
