import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# NLTK gereksinimlerini indir
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Performans için nesneler modül seviyesinde bir kez başlatılır
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

def process_text(text):
    """
    Ham müşteri yorumunu alır; küçük harfe çevirir, noktalama işaretlerini temizler,
    bağlaçları (stop-words) atar ve kelimeleri köklerine (lemmatization) indirger.
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)

def get_sentiment_score(text):
    """
    VADER kullanarak metnin duygu skorunu (sentiment score) -1 (negatif) ile +1 (pozitif) arasında hesaplar.
    """
    if pd.isna(text) or str(text).strip() == "":
        return 0.0
    return sia.polarity_scores(str(text))['compound']

def extract_word_count(text):
    """
    Bir metindeki toplam kelime sayısını döndürür.
    """
    return len(str(text).split())
