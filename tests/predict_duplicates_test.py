from src.app.duplicate_finder.predict import DuplicateFinder


def test_predict_duplicates():
    dummy_text = (
        "Bruchterme - gemeinsamer Nenner Bruchterme - gemeinsamer Nenner_1603916225648 "
        "Suche den gemeinsamen Nenner der beiden Bruchterme!"
    )

    duplicate_finder = DuplicateFinder()
    output = duplicate_finder.runByText(dummy_text, 0.8)

    assert len(output[0]) == 3
    assert dummy_text == output[0][2]
