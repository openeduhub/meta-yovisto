from fastapi import APIRouter
from pydantic import BaseModel
from starlette.status import HTTP_200_OK

from topic_finder.topic_assistant import TopicAssistant

router = APIRouter()


class TopicsInput(BaseModel):
    text: str


@router.post(
    "/topics",
    status_code=HTTP_200_OK,
)
def topics(input: TopicsInput):
    topic_assistant = TopicAssistant()
    return topic_assistant.go(input.text)
