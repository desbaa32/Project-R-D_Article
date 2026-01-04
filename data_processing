"""
Module de traitement des données pour la classification de sentiments
Inclut la collecte, le nettoyage et l'enrichissement des données
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Télécharger les ressources NLTK (à exécuter une seule fois)
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

class DataProcessor:
    """Classe pour le traitement des données sentiment140"""
    
    def __init__(self):
        self.scaler = None
        self.label_encoder = None
        self.tokenizer = None
    
    def load_data(self, filepath):
        """Charge le dataset Sentiment140"""
        print("Chargement du dataset...")
        df = pd.read_csv(filepath)
        print(f"Dataset chargé : {df.shape}")
        return df
    
    def explore_data(self, df):
        """Explore le dataset avec des statistiques"""
        print("\n=== Exploration des données ===")
        print(f"Shape : {df.shape}")
        print(f"Colonnes : {df.columns.tolist()}")
        print("\nValeurs manquantes :")
        print(df.isnull().sum())
        print("\nDistribution des sentiments :")
        print(df['sentiment'].value_counts())
        return df
    
    def preprocess_text(self, text, model_type='glove'):
        """Prétraite le texte selon le modèle cible"""
        if not isinstance(text, str):
            return ""
        
        # Nettoyage de base
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'([!?.])\1+', r'\1', text)
        
        if model_type == 'glove':
            # Pour GloVe : normalisation complète
            import unicodedata
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
            text = text.lower()
            text = re.sub(r'[^\w\s\']', ' ', text)
            
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
            text = ' '.join(tokens)
        
        elif model_type == 'bert':
            # Pour BERT : moins de prétraitement
            text = re.sub(r'[^\w\s@#]', ' ', text)
            text = ' '.join(text.split())
            
            if len(text.split()) > 512:
                text = ' '.join(text.split()[:510]) + ' [TRUNCATED]'
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def prepare_dataset(self, df, max_len=100, vocab_size=10000):
        """Prépare le dataset complet pour l'entraînement"""
        print("\n=== Préparation du dataset ===")
        
        # Prétraitement du texte
        df['text_glove'] = df['text'].apply(lambda x: self.preprocess_text(x, 'glove'))
        df['text_bert'] = df['text'].apply(lambda x: self.preprocess_text(x, 'bert'))
        
        # Supprimer les textes vides
        initial_len = len(df)
        df = df[(df['text_glove'].str.strip() != '') & (df['text_bert'].str.strip() != '')]
        print(f"Textes vides supprimés : {initial_len - len(df)}")
        
        # Conversion des labels
        if set(df['sentiment'].unique()) == {0, 4}:
            df['sentiment'] = df['sentiment'].replace({4: 1})
        
        # Encodage des variables catégorielles
        self.label_encoder = LabelEncoder()
        df['user_gender_encoded'] = self.label_encoder.fit_transform(df['user_gender'])
        df['user_location_encoded'] = self.label_encoder.fit_transform(df['user_location'])
        
        # Normalisation des variables numériques
        numerical_features = [
            'user_account_age_days', 'user_num_posts', 'user_num_followers',
            'user_num_retweets', 'tweet_hour', 'tweet_num_likes'
        ]
        
        self.scaler = StandardScaler()
        df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        
        # Split des données
        X_train, X_test = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['sentiment']
        )
        X_train, X_val = train_test_split(
            X_train, test_size=0.2, random_state=42, stratify=X_train['sentiment']
        )
        
        # Tokenization pour GloVe
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(X_train['text_glove'])
        
        # Préparer les séquences
        X_text_train = pad_sequences(
            self.tokenizer.texts_to_sequences(X_train['text_glove']),
            maxlen=max_len, padding='post'
        )
        X_text_val = pad_sequences(
            self.tokenizer.texts_to_sequences(X_val['text_glove']),
            maxlen=max_len, padding='post'
        )
        X_text_test = pad_sequences(
            self.tokenizer.texts_to_sequences(X_test['text_glove']),
            maxlen=max_len, padding='post'
        )
        
        # Attributs utilisateur
        user_attrs = [
            'user_gender_encoded', 'user_location_encoded', 'user_account_age_days',
            'user_num_posts', 'user_num_followers', 'user_num_retweets',
            'tweet_hour', 'tweet_num_likes'
        ]
        
        X_user_train = X_train[user_attrs].values
        X_user_val = X_val[user_attrs].values
        X_user_test = X_test[user_attrs].values
        
        # Labels
        y_train = X_train['sentiment'].values
        y_val = X_val['sentiment'].values
        y_test = X_test['sentiment'].values
        
        # Structure des données
        data_dict = {
            'text': {
                'train': X_text_train, 'val': X_text_val, 'test': X_text_test
            },
            'user_attrs': {
                'train': X_user_train, 'val': X_user_val, 'test': X_user_test,
                'feature_names': user_attrs, 'n_features': len(user_attrs)
            },
            'labels': {
                'train': y_train, 'val': y_val, 'test': y_test
            },
            'vocab_size': vocab_size,
            'max_len': max_len
        }
        
        print("Dataset préparé avec succès!")
        print(f"Train: {len(y_train)} échantillons")
        print(f"Validation: {len(y_val)} échantillons")
        print(f"Test: {len(y_test)} échantillons")
        
        return data_dict
