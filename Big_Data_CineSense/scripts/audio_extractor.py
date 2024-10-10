import os
import time
import moviepy.editor as mp
from multiprocessing import Pool, cpu_count
from threading import Thread
import asyncio
from concurrent.futures import ThreadPoolExecutor

def extract_audio_from_video(video_path):
    """
    Extract audio from a video file and save it as a WAV file.

    Args:
    video_path (str): The path to the video file.
    """
    try:
        video_dir, video_file = os.path.split(video_path)
        video_name = os.path.splitext(video_file)[0]
        audio_path = os.path.join(video_dir, f'{video_name}.wav' )

        video = mp.VideoFileClip(video_path)
        if video.audio is None:
            print(f"No audio track found in {video_file}")
            return
        
        video.audio.write_audiofile(audio_path)
        print(f"Successfully extracted audio from {video_file} to {audio_path}")
    
    except Exception as e:
        print(f'Failed to extract audio from {video_file}: {e}')
    finally:
        if 'video' in locals():
            video.close()

# get all video files recursively from folder
def get_all_video_files(folder):
    """
    Recursively get all video files (MP4 format) from a folder.
    
    Args:
    folder (str): The path to the folder to search.
    Returns:
    list: List of paths to all video files found.
    """
    video_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    return video_files        

def serial_extraction(video_folder):
    """
    Extract audio from all videos in a folder serially.

    Args:
    video_folder(str): The path to the folder containing video files.
    """
    start = time.perf_counter()
    video_files = get_all_video_files(video_folder)
    for video_file in video_files:
        extract_audio_from_video(video_file)
    end = time.perf_counter()
    print(f'Serial : {round(end - start, 2)} second(s)')

def multiprocessing_extraction(video_folder):
    """
    Extract audio from all videos in a folder using multiprocessing.

    Args:
    video_folder(str): The path to the folder containing video files.
    """    
    start = time.perf_counter()
    video_files = get_all_video_files(video_folder)
    with Pool(processes=cpu_count()) as pool:
        pool.map(extract_audio_from_video, video_files)
    end = time.perf_counter()
    print(f'Multiprocessing: {round(end - start, 2)} second(s)')

# threading extraction
def threading_extraction(video_folder):
    """
    Extract audio from all videos in a folder using threading.

    Args:
    video_folder(str): The path to the folder containing video files.
    """
    start = time.perf_counter()
    video_files = get_all_video_files(video_folder)
    threads = []
    for video_file in video_files:
        thread = Thread(target=extract_audio_from_video, args=(video_file,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    end = time.perf_counter()
    print(f'Threading: {round(end - start, 2)} second(s)')

async def async_extraction(video_folder):
    """
    Extract audio from all videos in a folder using asyncio.

    Args:
    video_folder(str): The path to the folder containing video files.
    """
    start = time.perf_counter()
    video_files = get_all_video_files(video_folder)
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        tasks = [loop.run_in_executor(executor, extract_audio_from_video, video_file) for video_file in video_files]
        await asyncio.gather(*tasks)
    end = time.perf_counter()
    print(f'Asyncio: {round(end - start, 2)} second(s)')

if __name__ == '__main__':
    video_folder = 'video_output'
    
    serial_extraction(video_folder)
    multiprocessing_extraction(video_folder)
    threading_extraction(video_folder)
    asyncio.run(async_extraction(video_folder))

