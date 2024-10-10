import os

def load_data(file_path = 'data/video_urls.txt'):
    """
    Load video URLs from a text file.

    Args:
    file_path (str): Path to the text file containing video URLs.
    Returns:
    list: List of video URLs.

    Raises:
    FileNotFoundError: if the file does not exist.
    IOError: if anI/O error occurs.
    Exception: for any unexpected errors.
    """
    try:
        with open(file_path, 'r') as file:
            urls = [line.strip() for line in file]
            return urls
    except FileNotFoundError:
        print(f'Error: The file {file_path} was not found.')
    except IOError:
        print(f'Error: An I/O error occured while trying to read {file_path}.')
    except Exception as e:
        print(f'An unexpected error occured: {e}.')

if __name__ == '__main__':
    urls = load_data()
    if urls is not None:
        print(f'Loaded data: {urls}')
    else:
        print("No data loaded.")

