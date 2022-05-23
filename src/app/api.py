from fastapi import APIRouter
from pydantic import BaseModel, Field
from starlette.status import HTTP_200_OK, HTTP_502_BAD_GATEWAY

from topic_finder.topic_assistant import TopicAssistant

router = APIRouter()


class TopicsInput(BaseModel):
    text: str


@router.post(
    "/topics",
    status_code=HTTP_200_OK,
    responses={HTTP_502_BAD_GATEWAY: {"description": "Topics service not responding"}},
)
def topics(topics_input: TopicsInput):
    topic_assistant = TopicAssistant()
    print(topic_assistant.tree.nodes)
    return topic_assistant.go(topics_input.text)


class Ping(BaseModel):
    status: str = Field(
        default="not ok",
        description="Ping output. Should be 'ok' in happy case.",
    )


@router.get(
    "/_ping",
    description="Ping function for automatic health check.",
    response_model=Ping,
)
def ping():
    return {"status": "ok"}
