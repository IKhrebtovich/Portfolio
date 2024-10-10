import os
from pathlib import Path
from nrclex import NRCLex
import spacy, nltk

# Load English tokenizer, tegger, parser, NER and word vectors
nlp = spacy.load('en_core_web_sm')

# Download necessary NLTK data
nltk.download('punkt')


def emotions_from_text(text_path):
    """
    Analyze emotions and frequencies from text file using NRC Lexicon.

    Args:
    text_path (str): The path to the text file to analyze.
    """
    try:
        video_dir, text_file = os.path.split(text_path)
        text_name = os.path.splitext(text_file)[0]
        output_path = os.path.join(video_dir, f'{text_name}_emotions.txt')

        with open(text_path, 'r', encoding = 'utf-8') as file:
            text = file.read()
            
            # Tokenize text using spaCy
            doc = nlp(text)

            #Join sentences into a full text string
            full_text = ' '.join([sent.text for sent in doc.sents])
            
            # Perform emotion analysis using NRC Lexicon
            emotion = NRCLex(text)
            
            with open(output_path, 'w', encoding = 'utf-8') as out_file:
                out_file.write(f'Detected Emotions and Frequencies: {emotion.affect_frequencies}')
                               
    except Exception as e:
        print(f'Error processing {text_file}: {e}')

def get_all_text_files(folder):
    """
    Recursively find all text files (excluding certain patterns) in a folder.
    
    Args:
    folder (str): The path to the folder to search.
    Returns:
    list: List of paths to all text files found.
    """
    text_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.txt') and not file.endswith('emotions.txt') and not file.endswith('sensitivity.txt') and not file.endswith('spanish.txt'):
                text_files.append(os.path.join(root, file))
    return text_files

if __name__ == '__main__':

    video_folder = 'video_output'
    text_files = get_all_text_files(video_folder)
    
    for text_file in text_files:
        emotions_from_text(text_file)