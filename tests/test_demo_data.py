from fake_news_detection.fake_news_detector import create_demo_dataset


def test_demo_data_has_balanced_binary_labels() -> None:
    data = create_demo_dataset()

    assert len(data) == 10
    assert set(data.columns) == {"title", "text", "source", "label"}
    assert data["label"].value_counts().to_dict() == {0: 5, 1: 5}
