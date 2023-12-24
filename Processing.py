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
            else:
                st.write("Pilih kolom teks terlebih dahulu.")

if __name__ == "__main__":
    main()


# Create a line chart using Matplotlib
st.write("Creating a line chart:")
st.line_chart(dF)  # This will create a line chart from the DataFrame

# Alternatively, using Matplotlib directly for more customization
st.write("Creating a customized line chart:")
plt.plot(df['X-Sentiment'], df['Y-Count'])  # Replace 'X-axis' and 'Y-axis' with your column names
st.pyplot()  # Display the Matplotlib plot in Streamlit

