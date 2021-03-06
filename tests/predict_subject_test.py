import os

from src.app.classification.predict import SubjectPredictor


def test_predict_subject():
    dummy_text = "Der Satz des Pythagoras lautet: a^2 + b^2 = c^2."

    model_file = "tests/data/wirlernenonline.oeh3.h5"
    label_file = "tests/data/wirlernenonline.oeh3.npy"
    tokenizer_file = "tests/data/wirlernenonline.oeh3.pickle"

    print(os.listdir(), os.getcwd())
    prediction = SubjectPredictor(model_file, label_file, tokenizer_file)
    assert prediction.tokenizer is not None
    result = prediction.run(dummy_text)
    assert len(result) == 1
    subject_is_mathematics = int(result[0][0]) == 380
    assert subject_is_mathematics
    probability_is_above_zero = result[0][1] > 0.5
    assert probability_is_above_zero
