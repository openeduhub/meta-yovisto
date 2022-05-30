from typing import Optional, Union

from fastapi import APIRouter
from pydantic import BaseModel, Field
from starlette.status import HTTP_200_OK, HTTP_502_BAD_GATEWAY
from topic_finder.topic_assistant import TopicAssistant, flatten_output

from app.classification.predict import SubjectPredictor
from app.duplicate_finder.predict import DuplicateFinder

router = APIRouter()


class WLOResponse(BaseModel):
    children: list[dict]
    data: dict[str, Union[str, int]]


class TopicsResponse(BaseModel):
    WLO: WLOResponse


class TopicsInput(BaseModel):
    text: str


@router.post(
    "/topics",
    status_code=HTTP_200_OK,
    # response_model=TopicsResponse,
    responses={HTTP_502_BAD_GATEWAY: {"description": "Topics service not responding"}},
)
def topics(topics_input: TopicsInput):
    topic_assistant = TopicAssistant()
    return flatten_output(topic_assistant.go(topics_input.text))


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


class PredictSubjectInput(BaseModel):
    text: str


@router.post(
    "/predict/subjects",
    status_code=HTTP_200_OK,
    responses={
        HTTP_502_BAD_GATEWAY: {"description": "Predict subject service not responding"}
    },
)
def predict_subject(prediction_input: PredictSubjectInput):
    modelFile = "data/wirlernenonline.oeh3.h5"
    labelFile = "data/wirlernenonline.oeh3.npy"
    tokenizerFile = "data/wirlernenonline.oeh3.pickle"

    prediction = SubjectPredictor(modelFile, labelFile, tokenizerFile)
    return prediction.run(prediction_input.text)


class PredictDuplicatesInput(BaseModel):
    id: Optional[str]
    url: Optional[str]
    text: Optional[str]
    threshold: Optional[float]


@router.post(
    "/predict/duplicates",
    status_code=HTTP_200_OK,
    response_model=list[list[Union[str, float]]],
    responses={
        HTTP_502_BAD_GATEWAY: {"description": "Predict subject service not responding"}
    },
)
def predict_duplicates(prediction_input: PredictDuplicatesInput):
    duplicate_finder = DuplicateFinder()

    if not prediction_input.threshold:
        threshold = 0.8
    else:
        threshold = prediction_input.threshold

    if prediction_input.id:
        output = duplicate_finder.runById(prediction_input.id)
    elif prediction_input.url:
        output = duplicate_finder.runByUrl(prediction_input.url)
    elif prediction_input.text:
        output = duplicate_finder.runByText(prediction_input.text, threshold)
    else:
        output = {}

    return output
