import os
import googleapiclient.discovery
from flask import Flask, request, jsonify
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')


app = Flask(__name__)

def get_comments(video_id, api_key):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    
    api_service_name = "youtube"
    api_version = "v3"
    
    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=api_key)

    comments = []
    
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )
    
    response = request.execute()
    
    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
            comments.append(comment)
        
        if 'nextPageToken' in response:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                pageToken=response['nextPageToken'],
                maxResults=100
            )
            response = request.execute()
        else:
            break

    return comments

def preprocess_comment(comment):
    comment = re.sub(r'[^a-zA-Z\s]', '', comment)
    comment = comment.lower()
    return comment

def classify_comments(comments):
    sid = SentimentIntensityAnalyzer()
    classified_comments = {'positive': [], 'neutral': [], 'negative': []}
    
    for comment in comments:
        scores = sid.polarity_scores(comment)
        if scores['compound'] >= 0.05:
            classified_comments['positive'].append(comment)
        elif scores['compound'] <= -0.05:
            classified_comments['negative'].append(comment)
        else:
            classified_comments['neutral'].append(comment)
    
    return classified_comments

@app.route('/analyze', methods=['POST'])
def analyze():
    video_id = request.json['video_id']
    api_key = "AIzaSyAdQav-iU4TAW9r9LIBYPl2l3-VcZjJslU"
    
    comments = get_comments(video_id, api_key)
    processed_comments = [preprocess_comment(comment) for comment in comments]
    classified_comments = classify_comments(processed_comments)
    
    return jsonify(classified_comments)

if __name__ == '__main__':
    app.run(debug=True)
