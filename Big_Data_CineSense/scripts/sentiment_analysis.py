import os
from pathlib import Path
from textblob import TextBlob
import nltk
from emotion_extractor import get_all_text_files


def extract_sensitivity_from_text(text_path):
    """
    Extract sentiment sensitivy from a text file using TextBlob.

    Args:
    text_path (str): The path to the text file to analyze.
    """
    try:
        video_dir, text_file = os.path.split(text_path)
        text_name = os.path.splitext(text_file)[0]
        output_path = os.path.join(video_dir, f'{text_name}_sensitivity.txt')
        
        with open(text_path, 'r', encoding = 'utf-8') as file:
            text = file.read()

            # Analyze sentiment using TextBlob
            blob = TextBlob(text)
            sentiment = blob.sentiment
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            with open(output_path, 'w', encoding = 'utf-8') as out_file:
                out_file.write(f'Sentiment: {sentiment}\n')
                out_file.write(f'Polarity: {polarity}\n')
                out_file.write(f'Subjectivity: {subjectivity}')
                
    except Exception as e:
        print(f'Error processing {text_file}: {e}')

if __name__ == '__main__':
    video_folder = 'video_output'
    text_files = get_all_text_files(video_folder)
    
    for text_file in text_files:
        extract_sensitivity_from_text(text_file)