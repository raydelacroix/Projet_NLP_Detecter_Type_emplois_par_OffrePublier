import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import xgboost as xgb
import nltk
import re
import joblib
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Téléchargement des ressources NLTK nécessaires
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')




class JobClassifier:
    def __init__(self, text_column='text', label_column='label', min_samples_per_class=50):
        """
        Initialisation du classificateur
        
        Args:
            text_column (str): Nom de la colonne contenant le texte
            label_column (str): Nom de la colonne contenant les étiquettes
            min_samples_per_class (int): Nombre minimum d'échantillons requis par classe
        """
        self.text_column = text_column
        self.label_column = label_column
        self.min_samples_per_class = min_samples_per_class
        self.tfidf = TfidfVectorizer(max_features=2000)
        self.le = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
    
    def filter_rare_classes(self, df):
        """Filtre les classes avec trop peu d'échantillons"""
        class_counts = df[self.label_column].value_counts()
        valid_classes = class_counts[class_counts >= self.min_samples_per_class].index
        filtered_df = df[df[self.label_column].isin(valid_classes)].copy()
        
        print(f"Nombre de classes avant filtrage: {len(class_counts)}")
        print(f"Nombre de classes après filtrage (min {self.min_samples_per_class} échantillons): {len(valid_classes)}")
        print("\nTop 10 classes les plus fréquentes après filtrage:")
        print(filtered_df[self.label_column].value_counts().head(10))
        
        return filtered_df
    
    def validate_dataframe(self, df):
        """Valide que le DataFrame contient les colonnes nécessaires"""
        missing_columns = []
        if self.text_column not in df.columns:
            missing_columns.append(self.text_column)
        if self.label_column not in df.columns:
            missing_columns.append(self.label_column)
        
        if missing_columns:
            raise ValueError(f"Colonnes manquantes dans le DataFrame: {', '.join(missing_columns)}")
    
    def preprocess_text(self, text):
        """Prétraitement du texte"""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('french'))
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)
    
    def prepare_data(self, df):
        """Préparation des données"""
        print("Préparation des données...")
        self.validate_dataframe(df)
        
        # Filtrage des classes rares
        filtered_df = self.filter_rare_classes(df)
        
        # Prétraitement
        filtered_df['processed_text'] = filtered_df[self.text_column].apply(self.preprocess_text)
        
        # Vectorisation et encodage
        X = self.tfidf.fit_transform(filtered_df['processed_text'])
        y = self.le.fit_transform(filtered_df[self.label_column])
        
        return X, y
    
    def initialize_models(self):
        """Initialisation des modèles avec des hyperparamètres simplifiés"""
        models_config = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [30],
                    'max_depth': [10],
                    'min_samples_split': [2]
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42),
                'params': {
                    'n_estimators': [30],
                    'max_depth': [3],
                    'learning_rate': [0.1]
                }
            }
        }
        return models_config
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Entraînement et optimisation des modèles"""
        models_config = self.initialize_models()
        best_score = 0
        
        print("\nEntraînement des modèles...")
        for model_name, config in models_config.items():
            print(f"\nEntraînement du modèle {model_name}...")
            
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=3,
                scoring='f1_weighted',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            self.models[model_name] = grid_search.best_estimator_
            
            score = grid_search.score(X_test, y_test)
            print(f"Meilleurs paramètres pour {model_name}: {grid_search.best_params_}")
            print(f"Score de validation pour {model_name}: {score:.3f}")
            
            if score > best_score:
                best_score = score
                self.best_model = grid_search.best_estimator_
                self.best_model_name = model_name
        
        print(f"\nMeilleur modèle: {self.best_model_name} avec un score de {best_score:.3f}")
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Évaluation d'un modèle"""
        y_pred = model.predict(X_test)
        
        print(f"\nRapport de classification pour {model_name}:")
        print(classification_report(y_test, y_pred, target_names=self.le.classes_))
        
        # Limiter le nombre de classes affichées dans la matrice de confusion
        n_classes_to_show = min(20, len(self.le.classes_))
        if len(self.le.classes_) > n_classes_to_show:
            print(f"\nAffichage de la matrice de confusion pour les {n_classes_to_show} premières classes...")
        
        cm = confusion_matrix(y_test, y_pred)[:n_classes_to_show, :n_classes_to_show]
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', 
                   xticklabels=self.le.classes_[:n_classes_to_show],
                   yticklabels=self.le.classes_[:n_classes_to_show])
        plt.title(f'Matrice de Confusion - {model_name}')
        plt.ylabel('Vraie classe')
        plt.xlabel('Classe prédite')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def evaluate_all_models(self, X_test, y_test):
        """Évaluation de tous les modèles"""
        print("\nÉvaluation des modèles:")
        for model_name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, model_name)
    
    def predict(self, text, model_name=None):
        """Prédiction pour un nouveau texte"""
        processed_text = self.preprocess_text(text)
        X = self.tfidf.transform([processed_text])
        
        if model_name and model_name in self.models:
            model = self.models[model_name]
        else:
            model = self.best_model
            model_name = self.best_model_name
        
        prediction = model.predict(X)
        probas = model.predict_proba(X)
        predicted_sector = self.le.inverse_transform(prediction)[0]
        class_probas = dict(zip(self.le.classes_, probas[0]))
        
        # Trier les probabilités par ordre décroissant et ne garder que les 5 plus élevées
        top_probas = dict(sorted(class_probas.items(), key=lambda x: x[1], reverse=True)[:5])
        
        return {
            'predicted_sector': predicted_sector,
            'model_used': model_name,
            'top_probabilities': top_probas
        }
    
    def save_models(self, filename_prefix):
        """Sauvegarde des modèles"""
        save_data = {
            'models': self.models,
            'tfidf': self.tfidf,
            'le': self.le,
            'best_model_name': self.best_model_name
        }
        joblib.dump(save_data, f'{filename_prefix}_complete.joblib')
        print(f"Modèles sauvegardés dans {filename_prefix}_complete.joblib")
    
    @classmethod
    def load_models(cls, filename):
        """Chargement des modèles"""
        instance = cls()
        save_data = joblib.load(filename)
        instance.models = save_data['models']
        instance.tfidf = save_data['tfidf']
        instance.le = save_data['le']
        instance.best_model_name = save_data['best_model_name']
        instance.best_model = instance.models[instance.best_model_name]
        return instance

if __name__ == "__main__":
    try:
        # Chargement des données
        df = pd.read_csv('C:/Users/user/Documents/Dossier Academique 2024/NLP/TP NLP/data_jobs.csv')
        print("Données chargées avec succès!")
        print(f"Nombre d'échantillons: {len(df)}")
        
        # Création du classifier avec filtre sur les classes rares
        classifier = JobClassifier(text_column='text', label_column='label', min_samples_per_class=50)
        
        # Préparation et division des données
        X, y = classifier.prepare_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entraînement et évaluation
        classifier.train_all_models(X_train, y_train, X_test, y_test)
        classifier.evaluate_all_models(X_test, y_test)
        
        # Exemples de prédictions
        exemples = [
            "Développeur Python avec expérience en machine learning",
            "Comptable senior pour banque internationale",
            "Infirmier urgentiste"
        ]
        
        print("\nExemples de prédictions:")
        for exemple in exemples:
            result = classifier.predict(exemple)
            print(f"\nTexte: {exemple}")
            print(f"Secteur prédit: {result['predicted_sector']}")
            print(f"Modèle utilisé: {result['model_used']}")
            print("Top 5 probabilités:")
            for sector, prob in result['top_probabilities'].items():
                print(f"  {sector}: {prob:.3f}")
        
        # Sauvegarde des modèles
        classifier.save_models('C:/Users/user/Documents/Dossier Academique 2024/NLP/TP NLP/job_classifier')
        
    except Exception as e:
        print(f"Erreur: {e}")