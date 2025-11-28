from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes

from fastapi import FastAPI
import uvicorn

from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="LLM API", version="1.0", description="Simple LLM API")

model = ChatGroq(model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Translate following into {language}"),
        ("human", "{text}"),
    ]
)

parser = StrOutputParser()

chain = prompt | model | parser

add_routes(app, chain, path='/chain')

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)