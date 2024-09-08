import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import datetime
import os

st.title('YouTube Comment Sentiment Analysis')

video_url = st.text_input('Enter YouTube Video URL')
def analyze_video(video_id):
    response = requests.post('http://127.0.0.1:5000/analyze', json={'video_id': video_id})
    if response.status_code == 200:
        return response.json()
    else:
        return None

if video_url:
    video_id = video_url.split('v=')[-1]
    result = analyze_video(video_id)

    if result:
        st.write('Positive Comments:')
        st.write(result['positive'])
        st.write('Neutral Comments:')
        st.write(result['neutral'])
        st.write('Negative Comments:')
        st.write(result['negative'])
        # Data Preparation
        sentiments = []
        sentiments.extend([('Positive', comment) for comment in result['positive']])
        sentiments.extend([('Neutral', comment) for comment in result['neutral']])
        sentiments.extend([('Negative', comment) for comment in result['negative']])
        
        sentiment_df = pd.DataFrame(sentiments, columns=['Sentiment', 'Comment'])
         # Count comments
        positive_count = len(result['positive'])
        neutral_count = len(result['neutral'])
        negative_count = len(result['negative'])

        # Display comment counts
        st.subheader('Comment Counts')
        st.write(f"Positive: {positive_count}")
        st.write(f"Neutral: {neutral_count}")
        st.write(f"Negative: {negative_count}")

        # Display pie chart
        st.subheader('Sentiment Distribution')
        labels = 'Positive', 'Neutral', 'Negative'
        sizes = [positive_count, neutral_count, negative_count]
        colors = ['#66b3ff', '#ffcc99', '#ff9999']
        explode = (0.1, 0, 0)

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

        # Display bar chart
        st.subheader('Sentiment Bar Chart')
        st.bar_chart({'Sentiment': ['Positive', 'Neutral', 'Negative'], 'Count': [positive_count, neutral_count, negative_count]})

        # Display word cloud
        st.subheader('Word Cloud')
        all_comments = ' '.join(result['positive'] + result['neutral'] + result['negative'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_comments)
        fig2, ax2 = plt.subplots()
        ax2.imshow(wordcloud, interpolation='bilinear')
        ax2.axis('off')
        st.pyplot(fig2)

        # Prepare data for trend analysis,  and histogram
        comments = result['positive'] + result['neutral'] + result['negative']
        sentiments = ['Positive'] * positive_count + ['Neutral'] * neutral_count + ['Negative'] * negative_count
        data = pd.DataFrame({'Comment': comments, 'Sentiment': sentiments})

        # Display sentiment trends over time
        st.subheader('Sentiment Trends Over Time')
        data['Sentiment Score'] = data['Comment'].apply(lambda x: 1 if x in result['positive'] else (-1 if x in result['negative'] else 0))
        data['Time'] = range(len(data))
        fig3, ax3 = plt.subplots()
        sns.lineplot(data=data, x='Time', y='Sentiment Score', ax=ax3)
        st.pyplot(fig3)
        # Display histogram
        st.subheader('Sentiment Histogram')
        fig5, ax5 = plt.subplots()
        sns.histplot(data=data, x='Sentiment', kde=True, ax=ax5)
        st.pyplot(fig5)

        # Save to CSV
        sentiment_df['Timestamp'] = datetime.datetime.now()
        csv_file = 'sentiment_analysis.csv'
        sentiment_df.to_csv(csv_file, index=False)
        
        st.write(f"Data saved to {csv_file}")
        st.write(f"File path: {os.path.abspath(csv_file)}")
            
            # Provide a download link
        with open(csv_file, "rb") as f:
                st.download_button(
                    label="Download CSV",
                    data=f,
                    file_name="sentiment_analysis.csv",
                    mime="text/csv"
                )
        
    else:
        st.write('Failed to fetch comments')
