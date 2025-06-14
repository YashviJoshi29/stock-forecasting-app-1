from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class FinBERTSentimentModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        self.model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        self.labels = ['negative', 'neutral', 'positive']

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]
        # Sentiment score: positive=1, neutral=0, negative=-1 (weighted sum)
        sentiment_score = float(np.dot(scores, [-1, 0, 1]))
        return sentiment_score

class SentimentAnalyzer:
    def __init__(self, model):
        self.model = model

    def analyze_sentiment(self, headlines):
        sentiment_scores = []
        for headline in headlines:
            score = self.model.predict(headline)
            sentiment_scores.append(score)
        return sentiment_scores

    def aggregate_sentiment(self, sentiment_scores):
        return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0