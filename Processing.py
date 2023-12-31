import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

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

# Main function
def main():
    st.title("Aplikasi Sentiment Analysis dengan Dataset CSV atau Excel")

    # Upload file
    uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                df = pd.read_excel(uploaded_file)
            else:
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

                    # Visualization - Bar Chart
                    sentiment_counts = df['Sentiment'].value_counts()
                    plt.figure(figsize=(8, 6))
                    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
                    plt.title('Distribusi Sentimen')
                    plt.xlabel('Sentimen')
                    plt.ylabel('Jumlah')
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    for index, value in enumerate(sentiment_counts.values):
                        plt.text(index, value + 1, str(value), ha='center', va='bottom')
                    st.pyplot()

                else:
                    st.write("Pilih kolom teks terlebih dahulu.")

        except Exception as e:
            st.write("Terjadi kesalahan:", e)

if __name__ == "__main__":
    main()
