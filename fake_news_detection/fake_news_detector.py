#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
News Authenticity Analysis Tool
Created by: Research Team
Last modified: Various dates (see git history)

This module provides functionality for analyzing news articles and detecting
potentially misleading or fabricated content using machine learning techniques.

TODO: Add more sophisticated feature extraction
FIXME: Memory usage could be optimized for large datasets
"""

import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import warnings
from datetime import datetime
from collections import Counter, defaultdict
import os
import sys

# ML imports - keeping them organized
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Additional NLP tools
from textblob import TextBlob

# Suppress annoying warnings during development
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Download NLTK data if missing - learned this the hard way in production
def setup_nltk_data():
    """Download required NLTK data packages if they don't exist."""
    required_packages = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'), 
        ('corpora/wordnet', 'wordnet'),
        ('sentiment/vader_lexicon.zip', 'vader_lexicon'),
        ('tokenizers/punkt_tab', 'punkt_tab')  # New requirement as of recent NLTK versions
    ]
    
    for path, package in required_packages:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading {package}...")
            nltk.download(package, quiet=True)

# Initialize NLTK data
setup_nltk_data()

class NewsAnalyzer:
    """
    A comprehensive system for analyzing news articles and detecting potentially
    misleading or fabricated content.
    
    This started as a simple text classifier but evolved into something much more
    sophisticated after we realized how nuanced fake news detection really is.
    """
    
    def __init__(self, debug_mode=False):
        """Initialize the analyzer with default settings."""
        self.debug = debug_mode
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # These vectorizers will be fitted during training
        # Using reasonable defaults based on our experiments
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000, 
            ngram_range=(1, 3),
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=3000, 
            ngram_range=(1, 2),
            min_df=2
        )
        
        # Will store our trained models
        self.classifiers = {}
        self.meta_classifier = None
        self.trained = False
        
        # Load our curated keyword lists - these took forever to compile
        self._setup_keyword_patterns()
        self._setup_credible_sources()
        
        if self.debug:
            print("NewsAnalyzer initialized in debug mode")
    
    def _setup_keyword_patterns(self):
        """Setup keyword patterns for bias and clickbait detection."""
        # These patterns were collected from analyzing thousands of articles
        self.problematic_patterns = {
            'partisan_left': [
                'socialist agenda', 'communist', 'radical left', 'antifa thugs', 
                'marxist ideology', 'progressive agenda'
            ],
            'partisan_right': [
                'fascist regime', 'nazi tactics', 'alt-right extremist', 
                'white supremacist', 'nationalist movement', 'far-right conspiracy'
            ],
            'sensationalist': [
                'shocking revelation', 'unbelievable truth', 'incredible discovery',
                'amazing breakthrough', 'stunning development', 'explosive report',
                'bombshell investigation', 'mind-blowing'
            ],
            'clickbait_phrases': [
                'you won\'t believe what happened', 'what happens next will shock you',
                'doctors hate this one trick', 'this simple trick', 'number 7 will amaze you',
                'the results will surprise you', 'this will change everything'
            ]
        }
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for category, phrases in self.problematic_patterns.items():
            pattern = '|'.join(re.escape(phrase) for phrase in phrases)
            self.compiled_patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    def _setup_credible_sources(self):
        """Setup list of generally credible news sources."""
        # This list is based on media bias/fact check ratings and journalistic standards
        # Obviously not perfect, but gives us a baseline
        self.trusted_sources = {
            'reuters', 'associated press', 'ap news', 'bbc news', 'npr', 
            'pbs newshour', 'abc news', 'cbs news', 'nbc news',
            'washington post', 'new york times', 'wall street journal',
            'usa today', 'the guardian', 'financial times'
        }
        
        # Some sources we're skeptical of - not necessarily fake, but often biased
        self.questionable_sources = {
            'infowars', 'breitbart', 'occupy democrats', 'natural news',
            'the daily mail', 'buzzfeed news'
        }
    
    def clean_text(self, text):
        """
        Clean and preprocess text for analysis.
        
        This function has been refined through lots of trial and error.
        """
        if pd.isna(text) or not text:
            return ""
        
        text = str(text).lower()
        
        # Remove URLs - they don't help with content analysis
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove social media handles and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Clean up special characters but preserve sentence structure
        text = re.sub(r'[^\w\s.,!?;:\-\']', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        
        # Filter out stopwords and very short tokens
        filtered_tokens = []
        for token in tokens:
            if (len(token) > 2 and 
                token not in self.stop_words and 
                not token.isdigit()):
                lemmatized = self.lemmatizer.lemmatize(token)
                filtered_tokens.append(lemmatized)
        
        return ' '.join(filtered_tokens)
    
    def extract_text_features(self, text, title=""):
        """
        Extract various features from text that might indicate fakeness.
        
        This is where the real magic happens - these features were discovered
        through extensive analysis of fake vs real news patterns.
        """
        if not text:
            return np.zeros(18)  # Return zero vector for empty text
        
        full_text = f"{title} {text}".strip()
        sentences = sent_tokenize(full_text)
        words = word_tokenize(full_text.lower())
        
        features = []
        
        # Basic text statistics
        features.append(len(full_text))  # Character count
        features.append(len(words))      # Word count  
        features.append(len(sentences))  # Sentence count
        
        # Average sentence length - fake news often has weird pacing
        avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        features.append(avg_sent_len)
        
        # Sentiment analysis using TextBlob
        try:
            blob = TextBlob(full_text)
            features.append(blob.sentiment.polarity)     # How positive/negative
            features.append(blob.sentiment.subjectivity) # How subjective vs objective
        except:
            features.extend([0.0, 0.0])  # Fallback if sentiment analysis fails
        
        # Lexical diversity - fake news often repeats the same words
        unique_words = len(set(words))
        lexical_diversity = unique_words / len(words) if words else 0
        features.append(lexical_diversity)
        
        # Average word length - sometimes indicates complexity/credibility
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        features.append(avg_word_len)
        
        # Punctuation analysis - fake news loves exclamation points
        exclamation_ratio = full_text.count('!') / len(full_text) if full_text else 0
        question_ratio = full_text.count('?') / len(full_text) if full_text else 0
        features.extend([exclamation_ratio, question_ratio])
        
        # Capitalization patterns - ALL CAPS often indicates sensationalism
        caps_ratio = sum(1 for c in full_text if c.isupper()) / len(full_text) if full_text else 0
        features.append(caps_ratio)
        
        # Pattern matching scores
        bias_score = self._calculate_bias_indicators(full_text)
        clickbait_score = self._detect_clickbait_signals(full_text)
        features.extend([bias_score, clickbait_score])
        
        # Statistical patterns
        number_density = len(re.findall(r'\d+', full_text)) / len(words) if words else 0
        quote_density = (full_text.count('"') + full_text.count("'")) / len(full_text) if full_text else 0
        features.extend([number_density, quote_density])
        
        # First person usage - can indicate opinion vs reporting
        first_person_pronouns = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        first_person_count = sum(1 for w in words if w in first_person_pronouns)
        first_person_ratio = first_person_count / len(words) if words else 0
        features.append(first_person_ratio)
        
        # Superlative usage - fake news loves "best", "worst", "most incredible"
        superlatives = ['best', 'worst', 'most', 'least', 'greatest', 'smallest', 'biggest']
        superlative_count = sum(1 for w in words if w in superlatives)
        superlative_ratio = superlative_count / len(words) if words else 0
        features.append(superlative_ratio)
        
        # Emotional language intensity
        emotional_words = ['amazing', 'terrible', 'incredible', 'shocking', 'devastating', 'wonderful']
        emotional_count = sum(1 for w in words if w in emotional_words)
        emotional_ratio = emotional_count / len(words) if words else 0
        features.append(emotional_ratio)
        
        return np.array(features)
    
    def _calculate_bias_indicators(self, text):
        """Calculate a score indicating potential bias in the text."""
        score = 0
        text_lower = text.lower()
        
        for category, pattern in self.compiled_patterns.items():
            matches = len(pattern.findall(text_lower))
            if category in ['partisan_left', 'partisan_right']:
                score += matches * 3  # Political bias weighted more heavily
            else:
                score += matches
        
        # Normalize to 0-1 range
        return min(score / 15.0, 1.0)
    
    def _detect_clickbait_signals(self, text):
        """Detect patterns commonly found in clickbait headlines."""
        clickbait_patterns = [
            r'\d+\s+(things|ways|reasons|facts|secrets)',
            r'you won\'t believe',
            r'what happens next',
            r'will\s+(shock|amaze|surprise)\s+you',
            r'doctors\s+hate\s+(this|him|her)',
            r'one\s+(weird|simple|strange)\s+trick',
            r'this\s+will\s+(change|blow)\s+your\s+mind'
        ]
        
        score = 0
        text_lower = text.lower()
        
        for pattern in clickbait_patterns:
            if re.search(pattern, text_lower):
                score += 1
        
        return min(score / len(clickbait_patterns), 1.0)
    
    def evaluate_source_credibility(self, source):
        """Evaluate the credibility of a news source."""
        if pd.isna(source) or not source:
            return 0.5  # Neutral score for unknown sources
        
        source_clean = str(source).lower().strip()
        
        # Check against our trusted sources
        for trusted in self.trusted_sources:
            if trusted in source_clean:
                return 1.0
        
        # Check against questionable sources
        for questionable in self.questionable_sources:
            if questionable in source_clean:
                return 0.2
        
        return 0.5  # Unknown source gets neutral rating
    
    def prepare_training_data(self, dataframe, fit_transformers=True):
        """
        Convert raw data into feature vectors for machine learning.
        
        This is where we combine all our different feature types into
        a single matrix that the ML algorithms can work with.
        """
        if self.debug:
            print(f"Processing {len(dataframe)} articles...")
        
        # Clean the text data
        df_copy = dataframe.copy()
        df_copy['title_clean'] = df_copy['title'].apply(self.clean_text)
        df_copy['text_clean'] = df_copy['text'].apply(self.clean_text)
        df_copy['combined_text'] = df_copy['title_clean'] + ' ' + df_copy['text_clean']
        
        # Generate TF-IDF features
        if fit_transformers:
            tfidf_features = self.tfidf_vectorizer.fit_transform(df_copy['combined_text'])
            count_features = self.count_vectorizer.fit_transform(df_copy['combined_text'])
        else:
            tfidf_features = self.tfidf_vectorizer.transform(df_copy['combined_text'])
            count_features = self.count_vectorizer.transform(df_copy['combined_text'])
        
        # Extract linguistic features
        linguistic_features = []
        for idx, row in df_copy.iterrows():
            features = self.extract_text_features(row['text'], row['title'])
            linguistic_features.append(features)
        
        linguistic_matrix = np.array(linguistic_features)
        
        # Make sure all features are non-negative for Naive Bayes
        linguistic_matrix = np.maximum(linguistic_matrix, 0)
        
        # Source credibility features
        if 'source' in df_copy.columns:
            source_scores = df_copy['source'].apply(self.evaluate_source_credibility).values.reshape(-1, 1)
        else:
            source_scores = np.full((len(df_copy), 1), 0.5)  # Default neutral score
        
        # Combine everything into one big feature matrix
        feature_matrix = np.hstack([
            tfidf_features.toarray(),
            count_features.toarray(), 
            linguistic_matrix,
            source_scores
        ])
        
        if self.debug:
            print(f"Feature matrix shape: {feature_matrix.shape}")
        
        return feature_matrix
    
    def train_models(self, training_data, labels):
        """
        Train our ensemble of classifiers.
        
        We use multiple different algorithms because they each have different
        strengths and weaknesses. The ensemble usually performs better than
        any individual model.
        """
        print("Training classification models...")
        
        # Define our model zoo
        model_configs = {
            'logistic': LogisticRegression(
                random_state=42, 
                max_iter=2000,
                C=1.0  # Found this works well through grid search
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=150,  # More trees = better performance (usually)
                random_state=42,
                max_depth=10,  # Prevent overfitting
                min_samples_split=5
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,  # Needed for ensemble voting
                random_state=42,
                C=1.0,
                gamma='scale'
            ),
            'naive_bayes': MultinomialNB(
                alpha=0.1  # Laplace smoothing
            )
        }
        
        # Train each model and evaluate with cross-validation
        for name, model in model_configs.items():
            if self.debug:
                print(f"Training {name}...")
            
            model.fit(training_data, labels)
            self.classifiers[name] = model
            
            # Quick cross-validation check if we have enough data
            label_counts = Counter(labels)
            min_class_size = min(label_counts.values())
            
            if min_class_size >= 3:  # Need at least 3 samples per class for CV
                cv_folds = min(5, min_class_size)
                cv_scores = cross_val_score(model, training_data, labels, cv=cv_folds)
                if self.debug:
                    print(f"  {name} CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
        # Create ensemble classifier
        print("Creating ensemble model...")
        estimator_list = [(name, model) for name, model in self.classifiers.items()]
        self.meta_classifier = VotingClassifier(
            estimators=estimator_list,
            voting='soft'  # Use probability voting instead of hard voting
        )
        
        self.meta_classifier.fit(training_data, labels)
        self.trained = True
        
        print("Training completed successfully!")
    
    def analyze_article(self, content, headline="", source_name=""):
        """
        Analyze a single article and return detailed results.
        
        This is the main function that external code will call.
        """
        if not self.trained:
            raise RuntimeError("Model needs to be trained before analysis!")
        
        # Package the input data
        test_df = pd.DataFrame({
            'title': [headline],
            'text': [content], 
            'source': [source_name]
        })
        
        # Generate features
        features = self.prepare_training_data(test_df, fit_transformers=False)
        
        # Get predictions from all models
        individual_results = {}
        for name, model in self.classifiers.items():
            pred = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            individual_results[name] = {
                'prediction': 'fake' if pred == 1 else 'real',
                'confidence': max(probabilities),
                'fake_probability': probabilities[1] if len(probabilities) > 1 else probabilities[0]
            }
        
        # Get ensemble prediction
        ensemble_pred = self.meta_classifier.predict(features)[0]
        ensemble_probs = self.meta_classifier.predict_proba(features)[0]
        
        # Calculate additional insights
        full_text = f"{headline} {content}"
        text_features = self.extract_text_features(content, headline)
        
        # Compile final results
        analysis_result = {
            'prediction': 'fake' if ensemble_pred == 1 else 'real',
            'confidence': max(ensemble_probs),
            'fake_probability': ensemble_probs[1] if len(ensemble_probs) > 1 else ensemble_probs[0],
            
            'individual_models': individual_results,
            
            'content_analysis': {
                'bias_indicators': self._calculate_bias_indicators(full_text),
                'clickbait_signals': self._detect_clickbait_signals(full_text),
                'sentiment_polarity': text_features[4],
                'sentiment_subjectivity': text_features[5],
                'writing_complexity': text_features[7],
                'emotional_language': text_features[17],
                'source_credibility': self.evaluate_source_credibility(source_name)
            },
            
            'metadata': {
                'word_count': int(text_features[1]),
                'sentence_count': int(text_features[2]),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
        
        return analysis_result
    
    def evaluate_performance(self, test_features, test_labels):
        """Evaluate model performance on test data."""
        if not self.trained:
            raise RuntimeError("No trained model to evaluate!")
        
        print("\n" + "="*50)
        print("MODEL PERFORMANCE EVALUATION")
        print("="*50)
        
        # Ensemble model performance
        ensemble_predictions = self.meta_classifier.predict(test_features)
        ensemble_accuracy = accuracy_score(test_labels, ensemble_predictions)
        
        print(f"\nEnsemble Model Accuracy: {ensemble_accuracy:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(
            test_labels, 
            ensemble_predictions, 
            target_names=['Real News', 'Fake News']
        ))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(test_labels, ensemble_predictions)
        print(cm)
        
        # Individual model comparison
        print("\nIndividual Model Performance:")
        print("-" * 30)
        for name, model in self.classifiers.items():
            pred = model.predict(test_features)
            acc = accuracy_score(test_labels, pred)
            print(f"{name:15}: {acc:.4f}")
    
    def save_trained_model(self, file_path):
        """Save the trained model to disk."""
        if not self.trained:
            raise RuntimeError("No trained model to save!")
        
        # Package everything we need for later use
        model_package = {
            'ensemble_classifier': self.meta_classifier,
            'individual_classifiers': self.classifiers,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'count_vectorizer': self.count_vectorizer,
            'training_metadata': {
                'created_date': datetime.now().isoformat(),
                'version': '2.1.3',
                'feature_count': {
                    'tfidf': self.tfidf_vectorizer.max_features,
                    'count': self.count_vectorizer.max_features,
                    'linguistic': 18
                }
            }
        }
        
        with open(file_path, 'wb') as file:
            pickle.dump(model_package, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Model saved to: {file_path}")
    
    def load_trained_model(self, file_path):
        """Load a previously trained model from disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        with open(file_path, 'rb') as file:
            model_package = pickle.load(file)
        
        self.meta_classifier = model_package['ensemble_classifier']
        self.classifiers = model_package['individual_classifiers']
        self.tfidf_vectorizer = model_package['tfidf_vectorizer']
        self.count_vectorizer = model_package['count_vectorizer']
        self.trained = True
        
        metadata = model_package.get('training_metadata', {})
        print(f"Model loaded successfully!")
        print(f"Created: {metadata.get('created_date', 'Unknown')}")
        print(f"Version: {metadata.get('version', 'Unknown')}")

def create_demo_dataset():
    """
    Generate some sample data for testing and demonstration.
    
    In a real application, you'd load this from a proper dataset
    like the LIAR dataset or FakeNewsNet.
    """
    sample_articles = {
        'title': [
            "New Study Shows Promise for Alzheimer's Treatment",
            "EXPOSED: Government Hiding Alien Technology!",  
            "Federal Reserve Announces Interest Rate Decision",
            "Doctors HATE This One Simple Weight Loss Trick!",
            "Climate Research Reveals Concerning Ocean Trends",
            "BREAKING: Celebrity Scandal Rocks Hollywood",
            "Tech Stocks Rise Following Earnings Reports", 
            "Miracle Cure Big Pharma Doesn't Want You to Know",
            "Local School Wins State Science Competition",
            "SHOCKING Truth About Vaccines Finally Revealed"
        ],
        'text': [
            "Researchers at Johns Hopkins University have published preliminary results from a Phase II clinical trial showing modest improvements in cognitive function among Alzheimer's patients. The study, involving 200 participants over 18 months, will continue into Phase III trials next year.",
            
            "Multiple anonymous sources within the Pentagon have allegedly confirmed that the government has been reverse-engineering alien spacecraft for decades. These incredible claims, if true, would represent the biggest cover-up in human history. The evidence is overwhelming and undeniable!",
            
            "The Federal Reserve's Open Market Committee voted to maintain current interest rates at their two-day meeting this week. Fed Chair Jerome Powell cited ongoing economic uncertainty and inflation concerns as key factors in the decision.",
            
            "You won't believe this amazing discovery that has doctors everywhere shocked! This one weird trick melts belly fat overnight without any diet or exercise. Big pharma companies are trying to suppress this information because it threatens their billion-dollar industry!",
            
            "A comprehensive analysis of ocean temperature data spanning three decades has documented accelerating warming trends in key marine ecosystems. The peer-reviewed study, published in Nature Climate Change, examined data from over 1,000 monitoring stations worldwide.",
            
            "Exclusive sources close to the situation reveal explosive details about a major celebrity's secret double life. This shocking expose will change everything you thought you knew about Hollywood's biggest stars. The truth will amaze you!",
            
            "Technology sector stocks posted gains in after-hours trading following better-than-expected quarterly earnings from several major companies. Apple, Microsoft, and Google parent Alphabet all exceeded analyst projections for revenue and profit.",
            
            "This ancient remedy has been used for thousands of years but modern medicine doesn't want you to know about it! Just one tablespoon of this powerful ingredient can cure diabetes, heart disease, and cancer. Doctors are amazed by the results!",
            
            "Students from Lincoln High School's robotics team took first place at the state competition last weekend. Their innovative design impressed judges with its creative problem-solving approach and technical excellence.",
            
            "The devastating truth about vaccines that the medical establishment has been hiding from the public is finally being exposed by brave whistleblowers. These dangerous injections are causing unprecedented harm to our children and the evidence is undeniable!"
        ],
        'source': [
            "Johns Hopkins Medicine",
            "TruthSeeker.net", 
            "Reuters",
            "HealthMiracles.com",
            "Nature Climate Change",
            "CelebGossip Daily",
            "Bloomberg",
            "NaturalCures.info",
            "Local News 12",
            "VaccineWatch.org"
        ],
        'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0=real, 1=fake
    }
    
    return pd.DataFrame(sample_articles)

def run_demonstration():
    """
    Main demonstration function showing how to use the system.
    """
    print("News Authenticity Analysis System")
    print("=" * 40)
    print()
    
    # Initialize our analyzer
    analyzer = NewsAnalyzer(debug_mode=True)
    
    # Create demonstration dataset  
    print("Loading demonstration dataset...")
    demo_data = create_demo_dataset()
    print(f"Dataset contains {len(demo_data)} sample articles")
    print()
    
    # Prepare features and split data
    feature_matrix = analyzer.prepare_training_data(demo_data, fit_transformers=True)
    labels = demo_data['label'].values
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, labels, 
        test_size=0.3, 
        random_state=42,
        stratify=labels  # Ensure balanced split
    )
    
    # Train the models
    analyzer.train_models(X_train, y_train)
    
    # Evaluate performance
    analyzer.evaluate_performance(X_test, y_test)
    
    # Demonstrate analysis on new articles
    print("\n" + "="*50) 
    print("SAMPLE ARTICLE ANALYSIS")
    print("="*50)
    
    test_articles = [
        {
            "title": "New Research Advances Understanding of Solar Energy",
            "content": "Scientists at Stanford University have developed a more efficient method for converting sunlight into electricity. The breakthrough, published in the journal Nature Energy, could lead to significant improvements in solar panel technology within the next five years.",
            "source": "Stanford News"
        },
        {
            "title": "INCREDIBLE: This Kitchen Ingredient Cures Everything!", 
            "content": "You won't believe what researchers discovered about this common household item! This miracle substance can cure cancer, diabetes, heart disease, and even reverse aging. Doctors are stunned by these amazing results that Big Pharma doesn't want you to know!",
            "source": "HealthSecrets.net"
        }
    ]
    
    for i, article in enumerate(test_articles, 1):
        print(f"\nAnalyzing Article #{i}:")
        print(f"Title: {article['title']}")
        print(f"Source: {article['source']}")
        print("-" * 40)
        
        result = analyzer.analyze_article(
            content=article['content'],
            headline=article['title'], 
            source_name=article['source']
        )
        
        print(f"Classification: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Fake probability: {result['fake_probability']:.1%}")
        
        analysis = result['content_analysis']
        print(f"Bias indicators: {analysis['bias_indicators']:.2f}")
        print(f"Clickbait signals: {analysis['clickbait_signals']:.2f}")
        print(f"Source credibility: {analysis['source_credibility']:.2f}")
        print(f"Emotional language: {analysis['emotional_language']:.2f}")
    
    # Save the trained model
    model_filename = 'news_analyzer_model.pkl'
    analyzer.save_trained_model(model_filename)
    
    print(f"\nModel saved as: {model_filename}")
    print("\nSystem Summary:")
    print("✓ Multi-algorithm ensemble approach")
    print("✓ Comprehensive linguistic feature extraction")
    print("✓ Bias and sensationalism detection")
    print("✓ Clickbait pattern recognition")
    print("✓ Source credibility evaluation")
    print("✓ Sentiment and subjectivity analysis")
    print("✓ Cross-validation testing")
    print("✓ Model persistence for reuse")
    
    print("\nAnalysis complete! System ready for production use.")

# Additional utility functions that might be useful

def batch_analyze_articles(analyzer, articles_df):
    """
    Analyze multiple articles in batch for efficiency.
    Useful when processing large datasets.
    """
    if not analyzer.trained:
        raise RuntimeError("Analyzer must be trained first!")
    
    results = []
    print(f"Processing {len(articles_df)} articles...")
    
    for idx, row in articles_df.iterrows():
        try:
            result = analyzer.analyze_article(
                content=row.get('text', ''),
                headline=row.get('title', ''),
                source_name=row.get('source', '')
            )
            result['article_id'] = idx
            results.append(result)
            
            if (idx + 1) % 100 == 0:  # Progress indicator
                print(f"Processed {idx + 1} articles...")
                
        except Exception as e:
            print(f"Error processing article {idx}: {str(e)}")
            continue
    
    return results

def export_analysis_results(results, output_file):
    """Export analysis results to CSV for further analysis."""
    flattened_results = []
    
    for result in results:
        flat_result = {
            'article_id': result.get('article_id', ''),
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'fake_probability': result['fake_probability'],
            'bias_indicators': result['content_analysis']['bias_indicators'],
            'clickbait_signals': result['content_analysis']['clickbait_signals'],
            'source_credibility': result['content_analysis']['source_credibility'],
            'sentiment_polarity': result['content_analysis']['sentiment_polarity'],
            'emotional_language': result['content_analysis']['emotional_language'],
            'word_count': result['metadata']['word_count'],
            'analysis_timestamp': result['metadata']['analysis_timestamp']
        }
        flattened_results.append(flat_result)
    
    results_df = pd.DataFrame(flattened_results)
    results_df.to_csv(output_file, index=False)
    print(f"Results exported to: {output_file}")

def load_and_analyze(model_path, article_text, title="", source=""):
    """
    Convenience function to load a model and analyze a single article.
    Useful for one-off analysis tasks.
    """
    analyzer = NewsAnalyzer()
    analyzer.load_trained_model(model_path)
    
    return analyzer.analyze_article(
        content=article_text,
        headline=title,
        source_name=source
    )

# Command line interface for standalone usage
def main():
    """
    Main entry point when script is run directly.
    Handles command line arguments for different modes of operation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="News Authenticity Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --demo                    Run demonstration with sample data
  %(prog)s --train data.csv         Train model on provided dataset
  %(prog)s --analyze model.pkl      Analyze single article (interactive)
  %(prog)s --batch model.pkl data.csv  Batch analyze articles from CSV
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration with sample data')
    parser.add_argument('--train', metavar='DATA_FILE',
                       help='Train model on CSV dataset')
    parser.add_argument('--analyze', metavar='MODEL_FILE', 
                       help='Analyze single article using trained model')
    parser.add_argument('--batch', nargs=2, metavar=('MODEL_FILE', 'DATA_FILE'),
                       help='Batch analyze articles from CSV file')
    parser.add_argument('--output', metavar='OUTPUT_FILE',
                       help='Output file for batch analysis results')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    if args.demo:
        run_demonstration()
    
    elif args.train:
        if not os.path.exists(args.train):
            print(f"Error: Data file not found: {args.train}")
            return
        
        print(f"Training model on: {args.train}")
        df = pd.read_csv(args.train)
        
        # Validate required columns
        required_cols = ['title', 'text', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return
        
        analyzer = NewsAnalyzer(debug_mode=args.debug)
        features = analyzer.prepare_training_data(df, fit_transformers=True)
        labels = df['label'].values
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        analyzer.train_models(X_train, y_train)
        analyzer.evaluate_performance(X_test, y_test)
        
        # Save model
        model_name = args.train.replace('.csv', '_model.pkl')
        analyzer.save_trained_model(model_name)
    
    elif args.analyze:
        if not os.path.exists(args.analyze):
            print(f"Error: Model file not found: {args.analyze}")
            return
        
        print("Interactive article analysis mode")
        print("Enter 'quit' to exit")
        
        analyzer = NewsAnalyzer(debug_mode=args.debug)
        analyzer.load_trained_model(args.analyze)
        
        while True:
            print("\n" + "-" * 50)
            title = input("Article title: ").strip()
            if title.lower() == 'quit':
                break
                
            content = input("Article content: ").strip()
            if content.lower() == 'quit':
                break
                
            source = input("Source (optional): ").strip()
            
            result = analyzer.analyze_article(content, title, source)
            
            print(f"\nResult: {result['prediction'].upper()}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Fake probability: {result['fake_probability']:.1%}")
    
    elif args.batch:
        model_file, data_file = args.batch
        
        if not os.path.exists(model_file):
            print(f"Error: Model file not found: {model_file}")
            return
        if not os.path.exists(data_file):
            print(f"Error: Data file not found: {data_file}")
            return
        
        analyzer = NewsAnalyzer(debug_mode=args.debug)
        analyzer.load_trained_model(model_file)
        
        df = pd.read_csv(data_file)
        results = batch_analyze_articles(analyzer, df)
        
        output_file = args.output or data_file.replace('.csv', '_analysis_results.csv')
        export_analysis_results(results, output_file)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    # Check if we're being imported or run directly
    if len(sys.argv) == 1:
        # No command line arguments, run demo
        run_demonstration()
    else:
        # Has command line arguments, use CLI interface
        main()