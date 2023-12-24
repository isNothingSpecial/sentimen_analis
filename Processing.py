import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud, STOPWORDS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

X = data['comment']
Y = data['sentimen']

nltk.download('stopwords')


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

X = X.apply(preprocessing)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)

y_pred = knn.predict(X_test)

# Main function
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
                
                # Create a line chart using Matplotlib
                st.write("Visualisasi Rekap Hasil Analisis:")
                sentiment_counts = df['Sentiment'].value_counts()
                st.bar_chart(sentiment_counts)  # This will create a line chart from sentiment counts
            else:
                st.write("Pilih kolom teks terlebih dahulu.")

if __name__ == "__main__":
    main()

accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred, average='weighted')
recall = recall_score(Y_test, y_pred, average='weighted')
f1 = f1_score(Y_test, y_pred, average='weighted')

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
