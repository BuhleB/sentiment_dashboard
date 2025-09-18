
import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import re
import nltk
import os

# Ensure NLTK data is available
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
if os.path.exists(nltk_data_path):
    nltk.data.path.append(nltk_data_path)

def analyze_sentiment_textblob(text):
    analysis = TextBlob(text)
    # Polarity ranges from -1.0 (negative) to 1.0 (positive)
    # Subjectivity ranges from 0.0 (objective) to 1.0 (subjective)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity

    if polarity > 0:
        sentiment = 'Positive'
    elif polarity < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    # Confidence score can be derived from the absolute polarity
    # Higher absolute polarity means higher confidence
    confidence = abs(polarity)

    return sentiment, confidence, polarity, subjectivity

def extract_keywords(text, num_keywords=5):
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        # Fallback to basic stop words if NLTK data is not available
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'])
    
    try:
        word_tokens = word_tokenize(text.lower())
    except LookupError:
        # Fallback to simple split if NLTK tokenizer is not available
        word_tokens = text.lower().split()
    
    # Filter out stop words and non-alphabetic tokens
    filtered_words = [word for word in word_tokens if word.isalpha() and word not in stop_words]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Get the most common words
    most_common = word_counts.most_common(num_keywords)
    
    return [word for word, count in most_common]

def batch_analyze_sentiment(texts):
    results = []
    for text_item in texts:
        sentiment, confidence, polarity, subjectivity = analyze_sentiment_textblob(text_item['text'])
        keywords = extract_keywords(text_item['text'])
        results.append({
            'text': text_item['text'],
            'sentiment': sentiment,
            'confidence': confidence,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'keywords': keywords,
            'source': text_item.get('source', 'N/A'),
            'date': text_item.get('date', 'N/A')
        })
    return pd.DataFrame(results)

def get_sentiment_explanation(text, sentiment, polarity):
    if sentiment == 'Positive':
        return f"The text '{text}' is classified as Positive due to words and phrases indicating approval or satisfaction. The polarity score is {polarity:.2f}."
    elif sentiment == 'Negative':
        return f"The text '{text}' is classified as Negative due to words and phrases indicating disapproval or dissatisfaction. The polarity score is {polarity:.2f}."
    else:
        return f"The text '{text}' is classified as Neutral. The polarity score is {polarity:.2f}. This could be due to a lack of strong emotional language or a balance of positive and negative terms."




