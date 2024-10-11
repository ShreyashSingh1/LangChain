import os
import uvicorn
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

# os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

llm = Ollama(model="llama3.2:1b")
prompt = ChatPromptTemplate.from_template("Write me an poem about {topic} with 100 words")

add_routes(
    app,
    prompt|llm,
    path="/poem"
)

# model=ChatOpenAI()
# prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")

# add_routes(
#     app,
#     prompt1|model,
#     path="/essay"
# )

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

