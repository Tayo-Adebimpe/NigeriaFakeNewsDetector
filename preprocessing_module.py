# preprocessing_module.py

import re
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize using wordpunct_tokenize (NO punkt dependency)
    tokens = wordpunct_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)
