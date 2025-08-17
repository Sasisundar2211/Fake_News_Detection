from .fake_news_detector import FakeNewsDetector, create_sample_dataset
from sklearn.model_selection import train_test_split

def main():
    print("=== ADVANCED FAKE NEWS DETECTION SYSTEM ===")

    # Step 1: Initialize the model
    detector = FakeNewsDetector()

    # Step 2: Load or create dataset
    print("\nLoading dataset...")
    df = create_sample_dataset()
    print(f"Dataset loaded with {len(df)} samples")

    # Step 3: Feature extraction
    print("\nExtracting features...")
    X = detector.prepare_features(df, fit_vectorizers=True)
    y = df["label"].values

    # Step 4: Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Step 5: Train the model
    print("\nTraining the models...")
    detector.train_models(X_train, y_train)

    # Step 6: Evaluate the model
    print("\nEvaluating the model...")
    detector.evaluate_model(X_test, y_test)

    # Step 7: Save the trained model
    print("\nSaving the trained model...")
    detector.save_model("fake_news_detector_model.pkl")

    # Step 8: Run prediction examples
    print("\n=== DEMO PREDICTIONS ===")

    test_cases = [
        {
            "title": "Scientists Make Breakthrough in Renewable Energy",
            "text": "MIT researchers have developed solar panels that are 15% more efficient than previous models. The study was peer-reviewed and published in Science.",
            "source": "MIT News"
        },
        {
            "title": "You Won't Believe What This Food Does!",
            "text": "This miracle food will cure all diseases! Doctors hate this one simple trick that pharmaceutical companies don't want you to know.",
            "source": "ClickBaitBlog.com"
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {case['title']}")
        result = detector.predict(text=case["text"], title=case["title"], source=case["source"])
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Fake Probability: {result['fake_probability']:.2%}")
        print(f"Bias Score: {result['analysis']['bias_score']:.2f}")
        print(f"Clickbait Score: {result['analysis']['clickbait_score']:.2f}")
        print(f"Source Credible: {'Yes' if result['analysis']['source_credible'] else 'No'}")

    print("\n🎉 System training complete. Model is saved and ready to use.")

if __name__ == "__main__":
    main()
