import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

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
                st.write("Creating a line chart:")
                sentiment_counts = df['Sentiment'].value_counts()
                st.line_chart(sentiment_counts)  # This will create a line chart from sentiment counts
            else:
                st.write("Pilih kolom teks terlebih dahulu.")

if __name__ == "__main__":
    main()

    
