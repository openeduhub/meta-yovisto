from typing import Union

from fastapi import APIRouter
from pydantic import BaseModel, Field
from starlette.status import HTTP_200_OK, HTTP_502_BAD_GATEWAY
from topic_finder.topic_assistant import TopicAssistant

from app.classification.predict import Prediction

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
    response_model=TopicsResponse,
    responses={HTTP_502_BAD_GATEWAY: {"description": "Topics service not responding"}},
)
def topics(topics_input: TopicsInput):
    topic_assistant = TopicAssistant()
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
    "/data/wirlernenonline.oeh3.h5 /data/wirlernenonline.oeh3.npy  /data/wirlernenonline.oeh3.pickle"
    modelFile = "data/wirlernenonline.oeh3.h5"
    labelFile = "data/wirlernenonline.oeh3.npy"
    tokenizerFile = "data/wirlernenonline.oeh3.pickle"

    prediction = Prediction(modelFile, labelFile, tokenizerFile)
    return prediction.run(prediction_input.text)


@router.post(
    "/train/subjects",
    status_code=HTTP_200_OK,
    responses={
        HTTP_502_BAD_GATEWAY: {"description": "Predict subject service not responding"}
    },
)
def train_subject(prediction_input: PredictSubjectInput):
    "/data/wirlernenonline.oeh3.h5 /data/wirlernenonline.oeh3.npy  /data/wirlernenonline.oeh3.pickle"
    modelFile = "/data/wirlernenonline.oeh3.h5"
    labelFile = "/data/wirlernenonline.oeh3.npy"
    tokenizerFile = "/data/wirlernenonline.oeh3.pickle"

    prediction = Prediction(modelFile, labelFile, tokenizerFile)
    print(prediction)
    return prediction.run(prediction_input.text)
