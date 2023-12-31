import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from pd import openpyxl

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
            elif uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
            else:
                st.write("Tipe file tidak didukung. Gunakan file CSV atau Excel.")

            st.write("Data yang Diunggah:")
            st.write(df)

            # Select column containing text data
            text_column = st.selectbox("Pilih kolom teks:", df.columns)

            # Analyze sentiment for selected text column
            if st.button("Analyze Sentiment") and text_column in df.columns:
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
                st.pyplot()

            elif st.button("Analyze Sentiment") and text_column not in df.columns:
                st.write("Kolom teks tidak ditemukan.")

        except Exception as e:
            st.write("Terjadi kesalahan:", e)

if __name__ == "__main__":
    main()
