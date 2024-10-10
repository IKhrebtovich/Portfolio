import os
from pathlib import Path
import speech_recognition as sr


def extract_text_from_audio(audio_path):
    """
    Extract text from audio file using Google's speach recognition API
    
    Args:
    audio_path (str): The path to the audio file (must be in WAV format).
    """
    try:
        video_dir, audio_file = os.path.split(audio_path)
        audio_name = os.path.splitext(audio_file)[0]
        text_path = os.path.join(video_dir, f'{audio_name}.txt')
        
        # Initialize the speech recognizer
        recognizer = sr.Recognizer()

        # Load audio file
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)

        # Perform speech recognition
        try:
            text = recognizer.recognize_google(audio)
            with open(text_path, 'w') as file:
                file.write(text)
            print(f'Text has been written to {text_path}')
        except sr.UnknownValueError:
            print(f'Could not understand audio in {audio_file}')
        except sr.RequestError as e:
            print(f'Error processing {audio_file}: {e}')
    except Exception as e:
        print(f'Error processing {audio_path}: {e}')

def get_all_audio_files(folder):
    """
    Recursively find all audio files (WAV format) in a folder.
    
    Args:
    folder (str): The path to the folder to search.
    Returns:
    list: List of paths to all audio files found.
    """
    audio_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    return audio_files


if __name__ == '__main__':
    video_folder = 'video_output'
    audio_files = get_all_audio_files(video_folder)
    
    for audio_file in audio_files:
        extract_text_from_audio(audio_file)