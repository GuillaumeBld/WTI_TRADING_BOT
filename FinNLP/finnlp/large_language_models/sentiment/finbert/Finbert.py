# File: trading_bot/FinNLP/large_language_models/sentiment/finbert/finbert.py

from transformers import pipeline

class FinBERT_Sentiment_Classif:
    """
    A FinBERT-based sentiment classifier that wraps the Hugging Face Transformers pipeline.
    """
    def __init__(self, model_name="ProsusAI/finbert", device=-1):
        """
        Initialize the classifier.

        Args:
            model_name (str): Model name to load (default "ProsusAI/finbert").
            device (int): Device index (-1 for CPU, 0 for GPU).
        """
        self.pipeline = pipeline(
            task="sentiment-analysis",
            model=model_name,
            device=device
        )

    def predict(self, texts):
        """
        Predict sentiment for the given text or list of texts.

        Args:
            texts (str or list): Input text or list of texts.

        Returns:
            list[dict]: A list of dictionaries with keys 'label' and 'score'.
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.pipeline(texts)