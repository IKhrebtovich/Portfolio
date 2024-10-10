import os
from pathlib import Path
from deep_translator import GoogleTranslator
import nltk
from emotion_extractor import get_all_text_files

def translate_text_to_spanish(text_path):   
    """
    Translates text from a file to Spanish using Google Translator API.

    Args:
    text_path (str): The path to the text file to translate.
    """
    try:
        video_dir, text_file = os.path.split(text_path)
        text_name = os.path.splitext(text_file)[0]
        output_path = os.path.join(video_dir, f'{text_name}_spanish.txt')
        
        with open(text_path, 'r', encoding = 'utf-8') as file:
            text = file.read()

            # Translate text to Spanish
            text_translated = GoogleTranslator(source='auto', target='es').translate(text)
            
            with open(output_path, 'w', encoding = 'utf-8') as out_file:
                out_file.write(text_translated)
                                
    except Exception as e:
            print(f'Error processing {text_file}: {e}')

if __name__ == '__main__':

    video_folder = 'video_output'
    text_files = get_all_text_files(video_folder)
    
    for text_file in text_files:
        translate_text_to_spanish(text_file)