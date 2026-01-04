"""
Module d'implémentation des modèles de classification de sentiments
TextCNN, UCRNN et UCRNN+ avec attention
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class PerformanceTracker:
    """Classe pour suivre les performances des modèles"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_training_timer(self):
        self.train_start_time = time.time()
    
    def stop_training_timer(self):
        self.train_time = time.time() - self.train_start_time
    
    def start_inference_timer(self):
        self.inference_start_time = time.time()
    
    def stop_inference_timer(self, n_samples):
        self.inference_time = time.time() - self.inference_start_time
        self.inference_time_per_sample = self.inference_time / n_samples
    
    def add_metrics(self, model_name, accuracy, precision, recall, f1, params_count):
        self.metrics[model_name] = {
            'training_time': self.train_time,
            'inference_time': self.inference_time,
            'inference_time_per_sample': self.inference_time_per_sample,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'params_count': params_count,
            'model_size_mb': params_count * 4 / (1024 * 1024)
        }


class ModelBuilder:
    """Classe pour construire et entraîner les modèles"""
    
    def __init__(self, vocab_size=10000, embedding_dim=100, max_len=100):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
    
    def load_glove_embeddings(self, tokenizer):
        """Charge les embeddings GloVe pré-entraînés"""
        print("Chargement des embeddings GloVe...")
        
        # Téléchargement des embeddings (version simplifiée)
        embeddings_index = {}
        try:
            import requests
            import io
            import zipfile
            
            # URL pour GloVe 6B 100d
            glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
            
            print("Téléchargement de GloVe...")
            response = requests.get(glove_url)
            zip_file = zipfile.ZipFile(io.BytesIO(response.content))
            
            # Lire le fichier d'embeddings
            with zip_file.open('glove.6B.100d.txt', 'r') as f:
                for line in f:
                    values = line.decode().split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                    
        except Exception as e:
            print(f"Erreur lors du chargement de GloVe: {e}")
            print("Utilisation d'embeddings aléatoires...")
            # En cas d'erreur, créer des embeddings aléatoires
            embedding_matrix = np.random.normal(size=(self.vocab_size, self.embedding_dim))
            return embedding_matrix
        
        # Créer la matrice d'embedding
        embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        words_found = 0
        
        for word, i in tokenizer.word_index.items():
            if i >= self.vocab_size:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                words_found += 1
        
        print(f"Matrice d'embedding créée: {embedding_matrix.shape}")
        print(f"Mots trouvés dans GloVe: {words_found}/{min(self.vocab_size, len(tokenizer.word_index))}")
        
        return embedding_matrix
    
    def build_textcnn(self, embedding_matrix=None):
        """Construit le modèle TextCNN"""
        model = keras.Sequential([
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_len,
                weights=[embedding_matrix] if embedding_matrix is not None else None,
                trainable=False if embedding_matrix is not None else True,
                name='embedding'
            ),
            layers.Conv1D(128, 5, activation='relu', padding='same', name='conv1d'),
            layers.GlobalMaxPooling1D(name='pooling'),
            layers.Dense(64, activation='relu', name='dense1'),
            layers.Dropout(0.5, name='dropout1'),
            layers.Dense(32, activation='relu', name='dense2'),
            layers.Dropout(0.3, name='dropout2'),
            layers.Dense(1, activation='sigmoid', name='output')
        ], name='TextCNN')
        
        return model
    
    def build_ucrnn(self, n_user_features, embedding_matrix=None):
        """Construit le modèle UCRNN original"""
        # Branche texte
        text_input = layers.Input(shape=(self.max_len,), name='text_input')
        
        text_embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            weights=[embedding_matrix] if embedding_matrix is not None else None,
            input_length=self.max_len,
            mask_zero=True,
            trainable=False,
            name='text_embedding'
        )(text_input)
        
        text_bilstm = layers.Bidirectional(
            layers.LSTM(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
            name='text_bilstm'
        )(text_embedding)
        
        text_features = layers.Dense(128, activation='relu', name='text_dense')(text_bilstm)
        text_features = layers.Dropout(0.5, name='text_dropout')(text_features)
        
        # Branche attributs utilisateur
        user_input = layers.Input(shape=(n_user_features,), name='user_input')
        user_normalized = layers.BatchNormalization(name='user_batchnorm')(user_input)
        user_reshaped = layers.Reshape((n_user_features, 1), name='user_reshape')(user_normalized)
        
        user_conv1 = layers.Conv1D(32, 3, activation='relu', padding='same', name='user_conv1')(user_reshaped)
        user_conv2 = layers.Conv1D(64, 3, activation='relu', padding='same', name='user_conv2')(user_conv1)
        user_pooled = layers.GlobalMaxPooling1D(name='user_pooling')(user_conv2)
        user_features = layers.Dense(64, activation='relu', name='user_dense')(user_pooled)
        user_features = layers.Dropout(0.5, name='user_dropout')(user_features)
        
        # Fusion
        merged = layers.Concatenate(name='feature_fusion')([text_features, user_features])
        merged_dense = layers.Dense(64, activation='relu', name='fusion_dense')(merged)
        merged_dense = layers.Dropout(0.5, name='fusion_dropout')(merged_dense)
        output = layers.Dense(1, activation='sigmoid', name='output')(merged_dense)
        
        model = keras.Model(
            inputs=[text_input, user_input],
            outputs=output,
            name='UCRNN'
        )
        
        return model
    
    def build_ucrnn_plus(self, n_user_features, embedding_matrix=None):
        """Construit le modèle UCRNN+ avec attention"""
        # Branche texte avec attention
        text_input = layers.Input(shape=(self.max_len,), name='text_input')
        
        text_embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            weights=[embedding_matrix] if embedding_matrix is not None else None,
            input_length=self.max_len,
            mask_zero=False,
            trainable=False,
            name='text_emb'
        )(text_input)
        
        text_bilstm = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            name='bilstm'
        )(text_embedding)
        
        # Attention simple
        attention_scores = layers.Dense(1, activation='tanh', name='att_dense')(text_bilstm)
        attention_scores = layers.Flatten(name='att_flatten')(attention_scores)
        attention_weights = layers.Activation('softmax', name='att_softmax')(attention_scores)
        
        attention_weights_repeated = layers.RepeatVector(256, name='att_repeat')(attention_weights)
        attention_weights_repeated = layers.Permute([2, 1], name='att_permute')(attention_weights_repeated)
        text_weighted = layers.Multiply(name='text_multiply')([text_bilstm, attention_weights_repeated])
        
        text_features = layers.GlobalAveragePooling1D(name='text_pool')(text_weighted)
        text_features = layers.Dense(128, activation='relu', name='text_dense1')(text_features)
        text_features = layers.Dropout(0.5, name='text_drop')(text_features)
        
        # Branche utilisateur
        user_input = layers.Input(shape=(n_user_features,), name='user_input')
        user_normalized = layers.BatchNormalization(name='user_norm')(user_input)
        user_reshaped = layers.Reshape((n_user_features, 1), name='user_reshape')(user_normalized)
        
        user_conv1 = layers.Conv1D(32, 3, activation='relu', padding='same', 
                                  kernel_regularizer=keras.regularizers.l2(0.001),
                                  name='user_conv1')(user_reshaped)
        user_conv2 = layers.Conv1D(64, 3, activation='relu', padding='same', name='user_conv2')(user_conv1)
        user_pooled = layers.GlobalMaxPooling1D(name='user_pool')(user_conv2)
        user_features = layers.Dense(64, activation='relu', name='user_dense')(user_pooled)
        user_features = layers.Dropout(0.5, name='user_dropout')(user_features)
        
        # Attention croisée
        text_proj = layers.Dense(32, activation='relu', name='text_proj')(text_features)
        user_proj = layers.Dense(32, activation='relu', name='user_proj')(user_features)
        
        attention_logits = layers.Dot(axes=1, normalize=True, name='cross_att')([text_proj, user_proj])
        attention_weights_cross = layers.Activation('sigmoid', name='cross_weights')(attention_logits)
        
        text_weighted_final = layers.Multiply(name='text_weight_final')([text_features, attention_weights_cross])
        user_weighted_final = layers.Multiply(name='user_weight_final')([
            user_features,
            layers.Lambda(lambda x: 1 - x, name='inv_weight')(attention_weights_cross)
        ])
        
        merged = layers.Concatenate(name='final_concat')([text_weighted_final, user_weighted_final])
        
        # Classification
        merged_dense = layers.Dense(48, activation='relu', name='dense1')(merged)
        merged_dense = layers.BatchNormalization(name='bn1')(merged_dense)
        merged_dense = layers.Dropout(0.3, name='drop1')(merged_dense)
        merged_dense = layers.Dense(24, activation='relu', name='dense2')(merged_dense)
        output = layers.Dense(1, activation='sigmoid', name='output')(merged_dense)
        
        model = keras.Model(
            inputs=[text_input, user_input],
            outputs=output,
            name='UCRNN_PLUS'
        )
        
        return model
    
    def train_model(self, model, data, model_name='Model', epochs=10, batch_size=64):
        """Entraîne un modèle et retourne l'historique"""
        print(f"\n=== Entraînement de {model_name} ===")
        
        # Compiler le modèle
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Entraînement
        if model_name == 'TextCNN':
            history = model.fit(
                data['text']['train'],
                data['labels']['train'],
                validation_data=(data['text']['val'], data['labels']['val']),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = model.fit(
                [data['text']['train'], data['user_attrs']['train']],
                data['labels']['train'],
                validation_data=(
                    [data['text']['val'], data['user_attrs']['val']],
                    data['labels']['val']
                ),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        return history, model
    
    def evaluate_model(self, model, data, model_name='Model', tracker=None):
        """Évalue un modèle et retourne les métriques"""
        print(f"\n=== Évaluation de {model_name} ===")
        
        if tracker:
            tracker.start_inference_timer()
        
        # Prédictions
        if model_name == 'TextCNN':
            y_pred_proba = model.predict(data['text']['test'], verbose=0)
        else:
            y_pred_proba = model.predict(
                [data['text']['test'], data['user_attrs']['test']],
                verbose=0
            )
        
        if tracker:
            tracker.stop_inference_timer(len(data['text']['test']))
        
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_true = data['labels']['test']
        
        # Métriques
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Paramètres: {model.count_params():,}")
        
        return accuracy, precision, recall, f1, y_pred
