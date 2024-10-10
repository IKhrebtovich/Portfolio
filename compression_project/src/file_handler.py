import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np

class FileHandler:
    """ 
    A utility class for file operations as reading, writing, and loading codebooks
    """

    @staticmethod
    def read_file(file_path, mode='rb'):
        """
        Reads the comtent of a file.

        Args:
            file_path (str): The path to the file to be read.
            mode (str): The mode in which to open the file. Defaults to 'rb' (read binary).

        Returns:
            bytes: The content of the file read into bytes.

        Raises:
            Exception: If an error occurs while reading the file, an error message is shown.
        """
        try:
            with open(file_path, mode) as file:
                return file.read()
        except Exception as e:
            messagebox.showerror("File read Error", f"An error occurred while reading the file: {str(e)}")
            return None
        
    @staticmethod    
    def write_file(file_path, data, mode='wb'):
        """
        Writes data to a file.

        Args:
            file_path (str): The path to the file where data will be written.
            data (bytes): The data to be written to the file.
            mode (str): The mode in which to open the file. Defaults to 'wb' (write binary).

        Returns:
            None

        Raises:
            Exception: If an error occurs while writing to the file, an error message is shown.
        """
        try:
            with open(file_path, mode) as file:
                file.write(data)
            return True
        except Exception as e:
            messagebox.showerror("File write Error", f"An error occurred while writing to the file: {str(e)}")
            return False
        
    @staticmethod
    def load_codebook_lzw(file_path):
        """
        Loads a codebook from a file.

        This method reads a codebook stored in a binary file and constructs a dictionary where each key is a sequence of bytes 
        and each value is an integer representing the encoded value. The file format is expected to be:
        - 1 byte for the length of the key.
        - N bytes for the key (where N is the length read in the previous step).
        - 4 bytes for the value (an integer).

        Args:
            file_path (str): The path to the file containing the codebook.

        Returns:
            dict: A dictionary where keys are byte sequences and values are integers.

        Raises:
            ValueError: If the file contains unexpected key or value lengths.

        Notes:
            - If the file ends before all data is read, it will be handled gracefully.
            - The key length is specified as a single byte, and values are assumed to be 4 bytes long.
        """
        codebook = {}
        try:
            with open(file_path, 'rb') as file:
                 while True:
                    # Read the length of the key (1 byte)
                    key_length_bytes = file.read(1)  
                    if not key_length_bytes:
                        break  # End of file
                    # Convert to int
                    key_length = int.from_bytes(key_length_bytes, byteorder='big')  
                    
                    # Read the key
                    key_bytes = file.read(key_length)  
                    if len(key_bytes) != key_length:
                        raise ValueError(f"Unexpected key length: {len(key_bytes)}")
                    
                    # Read the value (4 bytes)
                    value_bytes = file.read(4)  
                    if len(value_bytes) != 4:
                        raise ValueError(f"Unexpected value length: {len(value_bytes)}")
                    
                    # Convert to int
                    value = int.from_bytes(value_bytes, byteorder='big')

                    # Keep key as bytes  
                    codebook[key_bytes] = value  

        except Exception as e:
            raise ValueError(f"An error occured while loading the codebook: {str(e)}")
        return codebook

    @staticmethod
    def load_codebook_huffman(file_path):
        """
        Loads a Huffman codebook from a binary file.

        Args:
            file_path (str): Path to the codebook file.

        Returns:
            dict: A dictionary with keys as integers and values as binary strings.
        """    
        def bytes_to_bits(byte_data):
            """ Converts bytes to a binary string. """
            return ''.join(f'{byte:08b}' for byte in byte_data)  
                
        codebook = {}  
        try:
            with open(file_path, 'rb') as file:
                while True:
                    # Read key length and key
                    key_length_bytes = file.read(1)
                    if not key_length_bytes:
                        break     

                    key_length = int.from_bytes(key_length_bytes, byteorder='big')
                    key_bytes = file.read(key_length)
                    if len(key_bytes) != key_length:
                        raise ValueError(f'Incorrect key length: expected {key_length}, got {len(key_bytes)}')
                    key = int.from_bytes(key_bytes, byteorder='big')

                    # Read value length, value, and padding
                    value_length_bytes = file.read(1)
                    value_length = int.from_bytes(value_length_bytes, byteorder='big')
                    value_bytes = file.read(value_length)
                    if len(value_bytes) != value_length:
                        raise ValueError(f'Incorrect value length: expected {value_length}, got {len(value_bytes)}')
                
                    padding_length_bytes = file.read(1)
                    if not padding_length_bytes:
                        raise ValueError(f"Missing padding length")
                    
                    padding_length = int.from_bytes(padding_length_bytes, byteorder='big')

                    # Convert value bytes to binary and handle padding
                    value_bits = bytes_to_bits(value_bytes)
                    if padding_length > 0:
                        value_bits = value_bits[:-padding_length]
                   
                    codebook[key] = value_bits
            
        except Exception as e:
            raise ValueError(f"An error occurred while loading the codebook: {str(e)}")
    
        return codebook
    
       
    @staticmethod
    def read_image_file(file_path):
        """
        Reads the content of an image file.

        Args:
            file_path (str): The path to the image file to be read.

        Returns:
            numpy.ndarray: The image data as a NumPy array, or None if an error occurs.

        Raises:
            Exception: If an error occurs while reading the file, an error message is shown.
        """
        try:
            # Read the image file using OpenCV
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("The image file could not be loaded. Please check the file path and format.")
            
            return img

        except Exception as e:
            messagebox.showerror("File read Error", f"An error occurred while reading the file: {str(e)}")
            return None
    