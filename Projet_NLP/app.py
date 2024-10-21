import streamlit as st
import joblib
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Téléchargement des ressources NLTK nécessaires
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Chargement du modèle
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# Prétraitement du texte
def preprocess_text(text, lemmatizer):
    # Conversion en minuscules
    text = str(text).lower()
    
    # Suppression des caractères spéciaux
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Suppression des stopwords
    stop_words = set(stopwords.words('french'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Création du nuage de mots
def create_wordcloud(text):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        contour_width=3,
        contour_color='steelblue'
    ).generate(text)
    
    return wordcloud

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Classificateur d'Offres d'Emploi",
    page_icon="💼",
    layout="wide"
)

# Titre principal
st.title("🎯 Classificateur d'Offres d'Emploi")
st.markdown("---")

# Chargement des ressources
try:
    download_nltk_resources()
    model_data = load_model('job_classifier_complete.joblib')
    
    # Extraction des composants du modèle
    tfidf = model_data['tfidf']
    le = model_data['le']
    best_model = model_data['models'][model_data['best_model_name']]
    
    # Interface utilisateur
    st.subheader("📝 Saisissez votre offre d'emploi")
    job_text = st.text_area(
        "Description du poste",
        height=150,
        placeholder="Entrez la description du poste ici..."
    )
    
    if st.button("Analyser", type="primary"):
        if job_text.strip():
            # Création des colonnes pour l'affichage
            col1, col2 = st.columns([1, 1])
            
            with st.spinner("Analyse en cours..."):
                # Prétraitement
                lemmatizer = WordNetLemmatizer()
                processed_text = preprocess_text(job_text, lemmatizer)
                
                # Vectorisation
                X = tfidf.transform([processed_text])
                
                # Prédiction
                prediction = best_model.predict(X)
                probas = best_model.predict_proba(X)
                
                # Obtention des résultats
                predicted_class = le.inverse_transform(prediction)[0]
                class_probas = dict(zip(le.classes_, probas[0]))
                top_probas = dict(sorted(class_probas.items(), key=lambda x: x[1], reverse=True)[:5])
                
                # Affichage des résultats
                with col1:
                    st.subheader("🎯 Résultats de la classification")
                    st.markdown(f"**Catégorie prédite:** {predicted_class}")
                    
                    st.markdown("**Top 5 des probabilités:**")
                    for category, prob in top_probas.items():
                        prob_percentage = prob * 100
                        st.markdown(
                            f"- {category}: "
                            f"<div style='background-color:lightblue;width:{prob_percentage}%;padding:3px;'>"
                            f"{prob_percentage:.1f}%</div>",
                            unsafe_allow_html=True
                        )
                
                # Création et affichage du nuage de mots
                with col2:
                    st.subheader("☁️ Nuage de mots")
                    wordcloud = create_wordcloud(processed_text)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                
                # Affichage des mots-clés principaux
                st.markdown("---")
                st.subheader("🔑 Mots-clés principaux")
                words = processed_text.split()
                word_freq = pd.Series(words).value_counts().head(10)
                
                # Création d'un graphique à barres horizontal
                fig, ax = plt.subplots(figsize=(10, 5))
                word_freq.plot(kind='barh')
                plt.title("Top 10 des mots les plus fréquents")
                plt.xlabel("Fréquence")
                plt.ylabel("Mots")
                st.pyplot(fig)
                
        else:
            st.error("Veuillez entrer une description de poste à analyser.")
    
    # Ajout d'informations supplémentaires
    with st.expander("ℹ️ À propos de l'application"):
        st.markdown("""
        Cette application utilise un modèle d'apprentissage automatique pour classifier les offres d'emploi.
        Elle analyse le texte fourni et prédit la catégorie d'emploi la plus probable.
        
        **Fonctionnalités:**
        - Classification automatique des offres d'emploi
        - Visualisation des mots-clés sous forme de nuage
        - Analyse des fréquences des mots principaux
        - Affichage des probabilités pour les principales catégories
        """)

except Exception as e:
    st.error(f"Une erreur s'est produite lors du chargement du modèle: {str(e)}")
    st.info("Assurez-vous que le fichier 'job_classifier_complete.joblib' est présent dans le répertoire de l'application.")