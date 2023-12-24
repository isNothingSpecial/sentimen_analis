import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Function to analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        return "Positif"
    elif sentiment_score < 0:
        return "Negatif"
    else:
        return "Netral"

# Function to preprocess text
def preprocessing(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('indonesian'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return ' '.join(stemmed_tokens)

# Streamlit App
def main():
    st.title("Aplikasi Sentiment Analysis dengan Dataset CSV")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        st.write("Data yang Diunggah:")
        st.write(df)

        # Select column containing text data
        text_column = st.selectbox("Pilih kolom teks:", df.columns)

        # Analyze sentiment for selected text column
        if st.button("Analyze Sentiment"):
            if text_column:
                df['Sentiment'] = df[text_column].apply(analyze_sentiment)
                st.write("Hasil Analisis Sentimen:")
                st.write(df)
                
                # Create a bar chart using Streamlit
                st.write("Visualisasi Rekap Hasil Analisis:")
                sentiment_counts = df['Sentiment'].value_counts()
                st.bar_chart(sentiment_counts)  # This will create a bar chart from sentiment counts
                
                # Preprocess data for classification
                X = df[text_column].apply(preprocessing)
                Y = df['sentimen']

                vectorizer = TfidfVectorizer()
                X = vectorizer.fit_transform(X)

                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(X_train, Y_train)

                y_pred = knn.predict(X_test)

                # Evaluation metrics
                accuracy = accuracy_score(Y_test, y_pred)
                precision = precision_score(Y_test, y_pred, average='weighted')
                recall = recall_score(Y_test, y_pred, average='weighted')
                f1 = f1_score(Y_test, y_pred, average='weighted')

                st.write('Evaluasi Model:')
                st.write(f'Accuracy: {accuracy}')
                st.write(f'Precision: {precision}')
                st.write(f'Recall: {recall}')
                st.write(f'F1 Score: {f1}')
            else:
                st.write("Pilih kolom teks terlebih dahulu.")

if __name__ == "__main__":
    main()
