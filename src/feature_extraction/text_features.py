import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from scipy import stats
import re
from typing import Dict, List, Tuple

# Download required NLTK data
for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)

class TextFeatureExtractor:
    """
    Extracts comprehensive text features for depression detection.
    Features include linguistic, semantic, and psycholinguistic features.
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.depression_keywords = {
            'sad': ['sad', 'depressed', 'unhappy', 'miserable', 'gloomy'],
            'hopeless': ['hopeless', 'helpless', 'worthless', 'meaningless'],
            'suicidal': ['suicide', 'kill', 'die', 'dead', 'death'],
            'anxious': ['anxious', 'worried', 'nervous', 'scared', 'afraid'],
            'tired': ['tired', 'exhausted', 'fatigued', 'sleepy', 'weak'],
            'lonely': ['lonely', 'alone', 'isolated', 'abandoned'],
        }
    
    def extract(self, text: str) -> Dict[str, float]:
        """
        Extract all text features from input text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        # Clean and tokenize
        cleaned_text = self._clean_text(text)
        sentences = sent_tokenize(cleaned_text)
        words = word_tokenize(cleaned_text.lower())
        
        # Linguistic features
        features.update(self._extract_linguistic_features(text, sentences, words))
        
        # Sentiment features
        features.update(self._extract_sentiment_features(text))
        
        # Psycholinguistic features
        features.update(self._extract_psycholinguistic_features(text))
        
        # Depression indicators
        features.update(self._extract_depression_indicators(words))
        
        return features
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing special characters and normalizing."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s\.\?\!]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _extract_linguistic_features(self, text: str, sentences: List[str], 
                                    words: List[str]) -> Dict[str, float]:
        """Extract linguistic features."""
        features = {}
        
        if not words or not sentences:
            return {f'ling_{k}': 0.0 for k in ['word_count', 'sent_count', 'avg_word_len',
                                                'avg_sent_len', 'unique_words', 'lexical_diversity']}
        
        features['ling_word_count'] = float(len(words))
        features['ling_sent_count'] = float(len(sentences))
        features['ling_avg_word_len'] = np.mean([len(w) for w in words])
        features['ling_avg_sent_len'] = np.mean([len(word_tokenize(s)) for s in sentences])
        features['ling_unique_words'] = float(len(set(words)))
        features['ling_lexical_diversity'] = len(set(words)) / len(words) if words else 0.0
        
        # Punctuation features
        features['ling_exclamation_count'] = float(text.count('!'))
        features['ling_question_count'] = float(text.count('?'))
        features['ling_ellipsis_count'] = float(text.count('...'))
        
        # Part-of-speech features
        pos_tags = nltk.pos_tag(words)
        pos_counts = {}
        for word, pos in pos_tags:
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        features['ling_noun_ratio'] = pos_counts.get('NN', 0) / len(words) if words else 0.0
        features['ling_verb_ratio'] = pos_counts.get('VB', 0) / len(words) if words else 0.0
        features['ling_adj_ratio'] = pos_counts.get('JJ', 0) / len(words) if words else 0.0
        
        # Pronoun usage
        pronouns = {'i': 0, 'me': 0, 'my': 0, 'we': 0, 'us': 0, 'our': 0}
        for word in words:
            if word in pronouns:
                pronouns[word] += 1
        
        features['ling_first_person_singular'] = float(sum([pronouns['i'], pronouns['me'], pronouns['my']]))
        features['ling_first_person_plural'] = float(sum([pronouns['we'], pronouns['us'], pronouns['our']]))
        
        return features
    
    def _extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract sentiment features using TextBlob."""
        features = {}
        blob = TextBlob(text)
        
        features['sent_polarity'] = float(blob.sentiment.polarity)
        features['sent_subjectivity'] = float(blob.sentiment.subjectivity)
        
        # Sentiment over sentences
        sentiments = [TextBlob(sent).sentiment.polarity for sent in sent_tokenize(text)]
        if sentiments:
            features['sent_mean_polarity'] = float(np.mean(sentiments))
            features['sent_std_polarity'] = float(np.std(sentiments))
            features['sent_min_polarity'] = float(np.min(sentiments))
            features['sent_max_polarity'] = float(np.max(sentiments))
        else:
            features['sent_mean_polarity'] = 0.0
            features['sent_std_polarity'] = 0.0
            features['sent_min_polarity'] = 0.0
            features['sent_max_polarity'] = 0.0
        
        return features
    
    def _extract_psycholinguistic_features(self, text: str) -> Dict[str, float]:
        """Extract psycholinguistic features."""
        features = {}
        words = word_tokenize(text.lower())
        
        # Negative emotion words
        negative_words = ['bad', 'sad', 'angry', 'hate', 'terrible', 'awful', 
                         'horrible', 'disgusting', 'worse', 'worst']
        features['psyc_negative_words'] = float(sum(1 for w in words if w in negative_words))
        
        # Positive emotion words
        positive_words = ['good', 'great', 'happy', 'love', 'wonderful', 'excellent',
                         'amazing', 'fantastic', 'beautiful', 'best']
        features['psyc_positive_words'] = float(sum(1 for w in words if w in positive_words))
        
        # Intensity words
        intensifiers = ['very', 'really', 'extremely', 'absolutely', 'totally', 'so']
        features['psyc_intensifiers'] = float(sum(1 for w in words if w in intensifiers))
        
        # Negation words
        negations = ['not', 'no', 'never', 'neither', 'nobody', 'nothing']
        features['psyc_negations'] = float(sum(1 for w in words if w in negations))
        
        # Cognitive features
        cognitive_words = ['think', 'know', 'understand', 'believe', 'feel', 'want', 'need']
        features['psyc_cognitive_words'] = float(sum(1 for w in words if w in cognitive_words))
        
        return features
    
    def _extract_depression_indicators(self, words: List[str]) -> Dict[str, float]:
        """Extract depression-specific indicators."""
        features = {}
        
        for category, keywords in self.depression_keywords.items():
            count = sum(1 for w in words if w in keywords)
            features[f'dep_ind_{category}'] = float(count)
        
        return features


def create_text_feature_matrix(texts: List[str]) -> pd.DataFrame:
    """
    Create feature matrix from multiple texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        DataFrame with features as columns
    """
    extractor = TextFeatureExtractor()
    features_list = [extractor.extract(text) for text in texts]
    return pd.DataFrame(features_list)
