from typing import Optional, Union

from classification.predict import SubjectPredictor
from duplicate_finder.predict import DuplicateFinder
from fastapi import APIRouter
from pydantic import BaseModel, Field
from recommender.predict import Recommender
from starlette.status import HTTP_200_OK, HTTP_502_BAD_GATEWAY
from topic_finder.topic_assistant import TopicAssistant, flatten_children

router = APIRouter()


class WLOResponse(BaseModel):
    children: list[dict]
    data: dict[str, Union[str, int]]


class TopicsResponse(BaseModel):
    name: str = Field(
        description="Human readable name of the found topic with weight in brackets.",
    )
    weight: int = Field(
        description="Number of occurrences of this topic.",
    )
    uri: str = Field(
        description="URI of the found topic.",
    )
    match: str = Field(
        description="Additional information.",
    )
    label: str = Field(
        description="Name of the topic.",
    )


class TopicsInput(BaseModel):
    text: str


@router.post(
    "/topics",
    status_code=HTTP_200_OK,
    response_model=list[TopicsResponse],
    responses={HTTP_502_BAD_GATEWAY: {"description": "Topics service not responding"}},
    description="Predicts to which topics the input text could belong, e.g., to maths, calculus or analysis? "
    "Returns a list of objects with the uri's and names of matching topics according to SKOS.",
)
def topics(topics_input: TopicsInput):
    print("incoming")
    topic_assistant = TopicAssistant()
    print("outgoing")
    return flatten_children(topic_assistant.go(topics_input.text))


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
    description="Predicts the subjects, to which the given text belongs, e.g., mathematics or english.",
)
def predict_subject(prediction_input: PredictSubjectInput):
    model_file = "data/wirlernenonline.oeh3.h5"
    label_file = "data/wirlernenonline.oeh3.npy"
    tokenizer_file = "data/wirlernenonline.oeh3.pickle"

    prediction = SubjectPredictor(model_file, label_file, tokenizer_file)
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
    description="Predicts possible duplicates and returns their id, url and description based on input id, url or text."
    "Only one of the input elements is used, prioritized by id > url > text."
    "A threshold can be added as a minimum confidence required for a duplicate to be flagged as duplicate.",
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


class PredictSimilarDocumentsInput(BaseModel):
    id: str


@router.post(
    "/predict/similar_documents",
    status_code=HTTP_200_OK,
    response_model=list[list[Union[str, float]]],
    responses={
        HTTP_502_BAD_GATEWAY: {"description": "Predict subject service not responding"}
    },
    description="Predicts similar documents, which would be of interest to readers/consumers of this document.",
)
def predict_similar_documents(prediction_input: PredictSimilarDocumentsInput):
    model_file = "data/wirlernenonline.oeh-embed.h5"
    id_file = "data/wirlernenonline.oeh-id.pickle"
    duplicate_finder = Recommender(model_file, id_file)
    output = duplicate_finder.run(prediction_input.id)

    return output
