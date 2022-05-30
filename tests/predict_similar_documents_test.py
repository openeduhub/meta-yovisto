import os

from src.app.recommender.predict import Recommender


def test_predict_similar_documents():
    dummy_id = "1e6e21fd-e5d6-4046-a532-803708ea130d"

    model_file = "data/wirlernenonline.oeh-embed.h5"
    id_file = "data/wirlernenonline.oeh-id.pickle"

    print(os.listdir(), os.getcwd())
    prediction = Recommender(model_file, id_file)
    result = prediction.run(dummy_id)
    assert len(result) == 11
    last_id_is_known = result[0][0] == dummy_id
    assert last_id_is_known
    probability_is_one = result[0][1] == 1.0
    assert probability_is_one
    last_id_is_known = result[-1][0] == "b2798647-0e6b-421a-bd11-640f7dc807e5"
    assert last_id_is_known
    probability_is_below_one = round(result[-1][1], 2) == 0.97
    assert probability_is_below_one
