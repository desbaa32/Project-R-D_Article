"""
Script principal pour l'exécution du pipeline complet
Classification de sentiments avec modèles UCRNN
"""

import os
import sys
from data_processing import DataProcessor, download_nltk_resources
from model_implementation import ModelBuilder, PerformanceTracker
from visualization import ResultVisualizer

def main():
    """Fonction principale"""
    print("=" * 70)
    print("PROJET UCRNN - CLASSIFICATION DE SENTIMENTS")
    print("=" * 70)
    
    # Télécharger les ressources NLTK
    print("\n1. Préparation des ressources...")
    download_nltk_resources()
    
    # Initialisation des composants
    data_processor = DataProcessor()
    model_builder = ModelBuilder(vocab_size=10000, embedding_dim=100, max_len=100)
    tracker = PerformanceTracker()
    visualizer = ResultVisualizer()
    
    # Chemin vers les données
    data_path = "data/sentiment140_all_attributes.csv"
    
    if not os.path.exists(data_path):
        print(f"\nERREUR: Fichier de données non trouvé: {data_path}")
        print("Veuillez placer votre fichier CSV dans le dossier 'data/'")
        return
    
    # 1. Chargement et préparation des données
    print("\n2. Traitement des données...")
    df = data_processor.load_data(data_path)
    df = data_processor.explore_data(df)
    data = data_processor.prepare_dataset(df)
    
    # Charger les embeddings GloVe
    print("\n3. Chargement des embeddings...")
    embedding_matrix = model_builder.load_glove_embeddings(data_processor.tokenizer)
    
    # Variables pour stocker les résultats
    models = {}
    histories = []
    predictions = []
    model_names = []
    
    # 4. Modèle TextCNN (baseline)
    print("\n4. Construction et entraînement des modèles...")
    
    # TextCNN
    print("\n   a) TextCNN...")
    tracker.start_training_timer()
    textcnn = model_builder.build_textcnn(embedding_matrix)
    history_textcnn, textcnn = model_builder.train_model(
        textcnn, data, model_name='TextCNN', epochs=10, batch_size=64
    )
    tracker.stop_training_timer()
    
    acc, prec, rec, f1, y_pred = model_builder.evaluate_model(
        textcnn, data, model_name='TextCNN', tracker=tracker
    )
    tracker.add_metrics('TextCNN', acc, prec, rec, f1, textcnn.count_params())
    
    models['TextCNN'] = textcnn
    histories.append(history_textcnn)
    predictions.append(y_pred)
    model_names.append('TextCNN')
    
    # UCRNN
    print("\n   b) UCRNN...")
    tracker.start_training_timer()
    n_features = data['user_attrs']['n_features']
    ucrnn = model_builder.build_ucrnn(n_features, embedding_matrix)
    history_ucrnn, ucrnn = model_builder.train_model(
        ucrnn, data, model_name='UCRNN', epochs=10, batch_size=64
    )
    tracker.stop_training_timer()
    
    acc, prec, rec, f1, y_pred = model_builder.evaluate_model(
        ucrnn, data, model_name='UCRNN', tracker=tracker
    )
    tracker.add_metrics('UCRNN', acc, prec, rec, f1, ucrnn.count_params())
    
    models['UCRNN'] = ucrnn
    histories.append(history_ucrnn)
    predictions.append(y_pred)
    model_names.append('UCRNN')
    
    # UCRNN+
    print("\n   c) UCRNN+...")
    tracker.start_training_timer()
    ucrnn_plus = model_builder.build_ucrnn_plus(n_features, embedding_matrix)
    history_ucrnn_plus, ucrnn_plus = model_builder.train_model(
        ucrnn_plus, data, model_name='UCRNN+', epochs=10, batch_size=64
    )
    tracker.stop_training_timer()
    
    acc, prec, rec, f1, y_pred = model_builder.evaluate_model(
        ucrnn_plus, data, model_name='UCRNN+', tracker=tracker
    )
    tracker.add_metrics('UCRNN+', acc, prec, rec, f1, ucrnn_plus.count_params())
    
    models['UCRNN+'] = ucrnn_plus
    histories.append(history_ucrnn_plus)
    predictions.append(y_pred)
    model_names.append('UCRNN+')
    
    # 5. Visualisation des résultats
    print("\n5. Visualisation des résultats...")
    
    # Comparaison des performances
    df_comparison = visualizer.plot_performance_comparison(tracker)
    
    # Courbes d'apprentissage
    visualizer.plot_training_history(histories, model_names)
    
    # Matrices de confusion
    y_true = [data['labels']['test']] * 3  # Même vérité terrain pour tous
    visualizer.plot_confusion_matrices(y_true, predictions, model_names)
    
    # 6. Sauvegarde des résultats
    print("\n6. Sauvegarde des résultats...")
    
    # Sauvegarder le DataFrame de comparaison
    df_comparison.to_csv('results/model_comparison.csv', index=False)
    
    # Sauvegarder les modèles
    for name, model in models.items():
        model.save(f'models/{name.lower()}_model.h5')
    
    print("\n" + "=" * 70)
    print("PROJET TERMINÉ AVEC SUCCÈS!")
    print("=" * 70)
    print("\nRésultats sauvegardés dans:")
    print("  - results/model_comparison.csv")
    print("  - models/ (fichiers .h5)")
    print("\nVisualisations affichées à l'écran.")

if __name__ == "__main__":
    main()
