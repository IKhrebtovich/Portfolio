import os
import asyncio
from load_data import load_data
from download_video import test_parallel_process_executor
from audio_extractor import async_extraction
from text_extractor import extract_text_from_audio, get_all_audio_files
from translator import translate_text_to_spanish
from sentiment_analysis import extract_sensitivity_from_text
from emotion_extractor import emotions_from_text, get_all_text_files

def main():
    """
    Main function to ochestrate video downloading, audio extraction, text processing
    translation, sentiment and emotion analysis.
    """
    # Load video from video_urls.txt
    urls = load_data()
    print(f'Loaded data: {urls}')
    
    # Download and save each video in it's own folder
    print("Downloading and save video using parallel process executor...")
    test_parallel_process_executor(urls)

    video_folder = 'video_output'
    
    # Extract audio  
    print('Extracting audio using asyncio extraction...')
    asyncio.run(async_extraction(video_folder))

    # Process extracted audio files
    audio_files = get_all_audio_files(video_folder)
    for audio_file in audio_files:
        extract_text_from_audio(audio_file)
    
    # Process text files
    text_files = get_all_text_files(video_folder)
    for text_file in text_files:
        print(f'Translating {text_file} to Spanish')
        translate_text_to_spanish(text_file)

        print(f'Analyzing emotions in {text_file}')
        emotions_from_text(text_file)

        print(f'Analyzing sensitivity {text_file}')
        extract_sensitivity_from_text(text_file)

if __name__ == '__main__':
    main()