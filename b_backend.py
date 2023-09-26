import os
import openai
import PyPDF2
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext
from langchain.chat_models import ChatOpenAI
import streamlit as st


#os.environ["OPENAI_API_KEY"] = a_env_vars.OPENAI_API_KEY
#os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.getenv("OPENAI_API_KEY")

#Leer los PDFs
pdf = SimpleDirectoryReader('datos').load_data()

#Definir e instanciar el modelo
modelo = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'))

#Indexar el contenido de los PDFs
service_context = ServiceContext.from_defaults(llm_predictor=modelo)
index = GPTVectorStoreIndex.from_documents(pdf, service_context = service_context)

#Guardar el índice a disco para no tener que repetir cada vez
#Recordar que necesistaríamos persistir el drive para que lo mantenga
#index.save_to_disk('index.json')

#Cargar el índice del disco
#index = GPTVectorStoreIndex.load_from_disk('index.json')

def consulta(input_usuario):
    consulta = input_usuario + " Responde en español"
    resultado = index.as_query_engine().query(consulta)
    
    return resultado.response


