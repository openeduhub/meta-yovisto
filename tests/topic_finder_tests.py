from app.topic_finder.topic_assistant import TopicAssistant


def test_topic_finder():
    topic_assistant = TopicAssistant()
    assert topic_assistant.tree.nodes != {}
    dummy_text = (
        "Was ist OER?  Erklärvideos zu OER | OER Definition der UNESCO | Vorteile und Mehrwert von OER | "
        "OER Angebote in den Bildungsbereichen | wichtige offene Lizenzen"
    )
    found_topics = topic_assistant.go(dummy_text)
    assert len(found_topics.keys()) == 1
    assert found_topics["WLO"]["data"]["w"] == 7
