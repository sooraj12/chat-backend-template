from langchain_community.chat_models.ollama import ChatOllama

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

llm = ChatOllama(model="phi3")

origins = ["*"]

server = FastAPI(
    title="Server",
    version="1.0",
    description="A simple API server",
)

server.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@server.get("/health")
async def health():
    return {"status": "OK"}


class Data(BaseModel):
    query: str


def create_generator(query):
    print(query)
    for event in llm.stream(query):
        print(event.content)
        yield "data: " + event.content + "\n\n"


@server.post("/api/generate")
async def chat(data: Data):
    return StreamingResponse(
        create_generator(data.query), media_type="text/event-stream"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=server, host="localhost", port=8080)
