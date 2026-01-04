"""
Module de visualisation des résultats des modèles
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class ResultVisualizer:
    """Classe pour visualiser les résultats des modèles"""
    
    def __init__(self):
        # Configuration du style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
    
    def plot_performance_comparison(self, tracker):
        """Affiche la comparaison des performances des modèles"""
        if not tracker.metrics:
            print("Aucune métrique disponible")
            return
        
        # Créer un DataFrame pour la comparaison
        comparison_data = []
        for model_name, metrics in tracker.metrics.items():
            comparison_data.append({
                'Modèle': model_name,
                'Accuracy': metrics['accuracy'],
                'F1-Score': metrics['f1_score'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'Temps Entraînement (s)': metrics['training_time'],
                'Temps Inférence (ms)': metrics['inference_time_per_sample'] * 1000,
                'Paramètres': metrics['params_count']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Plot 1: Métriques de performance
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        colors = ['#FF6B6B', '#4ECDC4', '#FFA07A', '#98D8C8']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx//2, idx%2]
            bars = ax.bar(df['Modèle'], df[metric], color=colors[idx], alpha=0.8, edgecolor='black')
            ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12)
            ax.set_ylim([0.5, 1.0])
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Ajouter les valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Comparaison des Performances des Modèles', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Plot 2: Temps d'exécution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Temps d'entraînement
        bars1 = ax1.bar(df['Modèle'], df['Temps Entraînement (s)'], 
                       color='#FF6B6B', alpha=0.8, edgecolor='black')
        ax1.set_title('Temps d\'Entraînement', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Secondes', fontsize=12)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Temps d'inférence
        bars2 = ax2.bar(df['Modèle'], df['Temps Inférence (ms)'], 
                       color='#4ECDC4', alpha=0.8, edgecolor='black')
        ax2.set_title('Temps d\'Inférence par Échantillon', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Millisecondes', fontsize=12)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.suptitle('Comparaison des Temps d\'Exécution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return df
    
    def plot_training_history(self, histories, model_names):
        """Affiche les courbes d'apprentissage"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for idx, (history, name) in enumerate(zip(histories, model_names)):
            color = colors[idx % len(colors)]
            
            # Courbe de loss
            axes[0].plot(history.history['loss'], label=f'{name} - Train', 
                        color=color, linewidth=2, marker='o', markersize=4)
            if 'val_loss' in history.history:
                axes[0].plot(history.history['val_loss'], label=f'{name} - Val',
                           color=color, linewidth=2, linestyle='--', marker='s', markersize=4)
            
            # Courbe d'accuracy
            axes[1].plot(history.history['accuracy'], label=f'{name} - Train',
                        color=color, linewidth=2, marker='o', markersize=4)
            if 'val_accuracy' in history.history:
                axes[1].plot(history.history['val_accuracy'], label=f'{name} - Val',
                           color=color, linewidth=2, linestyle='--', marker='s', markersize=4)
        
        axes[0].set_title('Courbe de Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title('Courbe d\'Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].legend(loc='lower right')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Courbes d\'Apprentissage', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrices(self, y_true_list, y_pred_list, model_names):
        """Affiche les matrices de confusion pour chaque modèle"""
        from sklearn.metrics import confusion_matrix
        
        n_models = len(model_names)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (y_true, y_pred, name) in enumerate(zip(y_true_list, y_pred_list, model_names)):
            cm = confusion_matrix(y_true, y_pred)
            
            # Normalisation
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Heatmap
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                       cbar=False, square=True, ax=axes[idx])
            
            axes[idx].set_title(f'Matrice de Confusion - {name}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Prédit', fontsize=10)
            axes[idx].set_ylabel('Réel', fontsize=10)
            axes[idx].set_xticklabels(['Négatif', 'Positif'])
            axes[idx].set_yticklabels(['Négatif', 'Positif'])
        
        plt.suptitle('Matrices de Confusion', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
