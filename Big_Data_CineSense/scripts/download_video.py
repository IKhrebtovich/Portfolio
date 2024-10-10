from pytube import YouTube, exceptions
import os, time
from load_data import load_data
from datetime import datetime
from multiprocessing import Pool
import concurrent.futures
import multiprocessing
import threading
import re


# Mutex for thread safety
lock = threading.Lock()

def sanitize_filename(filename):
    """
    Sanitize a filename by removing or replacing invalid characters.

    Args:
    filename (str): The filename to sanitize.
    Returns:
    str: The sanitized filename.
    """
    return re.sub(r'[\\/*?:"<>|]',"", filename)

def downloading_and_save_video(url, semaphore = None):
    """
    Download video from a given URL and save it to the output directory

    Args:
    url (str): The URL of the YouTube video.
    semaphore(threading.Semaphore, optional): Semaphore to limit concurrent downloads.
    Returns:
    str: The output directory where the video is saved, or None if download fails.
    """
    try:
        yt = YouTube(url)
        folder_name = sanitize_filename(yt.title)
        output_dir = os.path.join('video_output', folder_name)
        os.makedirs(output_dir, exist_ok=True)
        stream = yt.streams.get_highest_resolution()
        print(f'Downloading video: {yt.title}')
        video_file_path = os.path.join(output_dir, f'{folder_name}.mp4')
        stream.download(output_path = output_dir, filename=f'{folder_name}.mp4')
        log_message = create_log_message(url, True)
        write_to_log(log_message)
       # print(f"Download completed: {yt.title}")
        return output_dir
    except exceptions.AgeRestrictedError:
        print(f"Video is age restricted and cannot be downloaded: {url}")
        log_message = create_log_message(url, False)
        write_to_log(log_message)     
    except Exception as e:
        print(f"An error occured while downloading video: {url}, Error: {e}")
        log_message = create_log_message(url, False)
        write_to_log(log_message) 
    finally:
        if semaphore:
            semaphore.release()

def create_log_message(url, download_status):
    """
    Create a log message for a video download.

    Args:
    url (str): The URL of the YouTube video.
    download_status (bool): Whether the download was successful.
    Returns:
    str: The formatted log message.
    """
    timestamp = datetime.now().strftime('%H:%M, %d %B %Y')
    thread_id = threading.get_ident() # get the thread ID
    process_id = os.getpid() # get the process ID
    return f'"Timestamp": {timestamp}, "URL": "{url}", "Download": {download_status}, "Thread ID": {thread_id}, "Process ID": {process_id} \n'

def write_to_log(message):
    """
    Write a log message to the download log file.

    Args: 
    message (str): The log message to write.
    """
    log_file = os.path.join('data', 'download_log.txt')
    with lock:
        try:
            with open(log_file, "a", encoding = 'utf-8') as f:
                f.write(message)
        except Exception as e:
            print(f'Error writing to log file: {log_file}, Error: {e}')

def download_videos_in_threads(urls):
    """
    Download videos using threads with a semaphore to limit downloads.

    Args: 
    urls (list): List of video URLs to download.
    """
    threads = []
    semaphore = threading.Semaphore(5)
    for url in urls:
        semaphore.acquire()
        thread = threading.Thread(target = downloading_and_save_video, args=(url,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

def test_serial_runner(urls):
    """
    Test serial download of videos.

    Args: 
    urls (list): List of video URLs to download.
    """
    if not urls:
        print('No URLs to process')
        return
    start = time.perf_counter()
    for url in urls:
        downloading_and_save_video(url, None)
    end = time.perf_counter()
    print(f'Serial: {round(end-start,2)} second(s)')

def test_parallel_process_executor(urls):
    """
    Test parallel download of videos using ProcessPoolExecutor.

    Args: 
    urls (list): List of video URLs to download.
    """
    if not urls:
        print('No URLs to process')
        return
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(downloading_and_save_video, urls)
    end = time.perf_counter()
    print(f'Parallel (ProcessPoolExecutor): {round(end-start,2)} second(s)')

def test_parallel_pool(urls):
    """
    Test parallel download of videos using multiprocessing Pool.

    Args: 
    urls (list): List of video URLs to download.
    """
    if not urls:
       print('No URLs to process')
       return
    start = time.perf_counter()
    with multiprocessing.Pool() as p:
        p.map(downloading_and_save_video, urls)
    end = time.perf_counter()
    print(f'Parallel (Pool): {round(end-start,2)} second(s)')

def test_thread_with_semaphore(urls):
    """
    Test parallel download of videos using threads with a semaphore.

    Args: 
    urls (list): List of video URLs to download.
    """
    if not urls:
        print('No URLs to process')
        return
    semaphore = threading.Semaphore(5)
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for url in urls:
            semaphore.acquire()
            executor.submit(downloading_and_save_video, url, semaphore)
    end = time.perf_counter()
    print(f'Parallel (Thread with semaphore): {round(end-start,2)} second(s)')

if __name__ == '__main__':
    urls = load_data()
    if urls:
       print('Testing serial runner...')
       test_serial_runner(urls)
       
       print('Testing parallel process executor...')
       test_parallel_process_executor(urls)
       
       print('Testing parallel pool...')
       test_parallel_pool(urls)
       
       print('Testing thread with semaphore...') 
       test_thread_with_semaphore(urls)
    else:
       print('No URLs found in the file to process.')
    






