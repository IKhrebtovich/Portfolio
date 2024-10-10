import unittest
import numpy as np
import os
from tkinter import ttk, filedialog, messagebox
import tkinter as tk
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.file_handler import FileHandler

class TestFileHandler(unittest.TestCase):

    def tearDown(self):
        """Remove test file if it exists."""
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def setUp(self):
        """Set up the test environment."""
        self.test_file_path = 'test_file.bin'
        self.test_data = b'Hello world!'
        self.file_handler = FileHandler()

        with open(self.test_file_path, 'wb') as f:
            f.write(self.test_data)

    def test_read_file(self):
        """Test reading from file."""
        data = FileHandler.read_file(self.test_file_path)
        self.assertEqual(data, self.test_data)

    @patch('src.file_handler.messagebox.showerror')
    def test_read_file_exception(self, mock_showerror):
        """Test handling exceptions when reading from file."""
        # Make the file path invalid to trigger an exception
        invalid_path = 'invalid_file_path.bin'
        
        result = FileHandler.read_file(invalid_path)
        
        # Check if error messagebox was shown
        self.assertIsNone(result)
        self.assertTrue(mock_showerror.called)

    def test_write_file(self):
        """Test writing to file."""
        write_data = b'Test data'
        FileHandler.write_file(self.test_file_path, write_data)

        read_data = FileHandler.read_file(self.test_file_path)
        self.assertEqual(read_data, write_data)
       
    def test_load_codebook_lzw(self):
        """Test loading LZW codebook from file."""
        codebook = {b'\x00\x01': 256, b'\x01\x02': 257}
        with open(self.test_file_path, "wb") as f:
            for key, value in codebook.items():
                f.write(len(key).to_bytes(1, 'big'))
                f.write(key)
                f.write(value.to_bytes(4, 'big'))
            
        loaded_codebook = FileHandler.load_codebook_lzw(self.test_file_path)
        self.assertEqual(loaded_codebook, codebook)

    def test_load_codebook_with_incorrect_key_length_lzw(self):
        """Test loading LZW codebook with incorrect key length."""
        with open(self.test_file_path, "wb") as f:
            f.write((2).to_bytes(1, 'big'))
            f.write(b'\x00')

        with self.assertRaises(ValueError):
             FileHandler.load_codebook_lzw(self.test_file_path)

    def test_load_codebook_with_incorrect_value_length_lzw(self):
        """Test loading LZW codebook with incorrect value length.""" 
        with open(self.test_file_path, "wb") as f:
            f.write((1).to_bytes(1, 'big'))
            f.write(b'\x00')
            f.write((257).to_bytes(2, 'big'))

        with self.assertRaises(ValueError):
            FileHandler.load_codebook_lzw(self.test_file_path) 

    def test_load_codebook_lzw_empty_file(self):
        """Test loading LZW codebook from an empty file."""
        open(self.test_file_path, 'wb').close()

        loaded_codebook = FileHandler.load_codebook_lzw(self.test_file_path)
        self.assertEqual(loaded_codebook, {})
    
    def bits_to_bytes(self, bits):
            """ Converts a binary string to bytes with padding. """
            bits_length = len(bits)
            padded_bits = bits.ljust((bits_length + 7) // 8 * 8, '0')
            padding_length = len(padded_bits) - bits_length
            byte_array = bytearray()
            for i in range(0, len(padded_bits), 8):
                byte = padded_bits[i:i+8]
                byte_array.append(int(byte, 2))
            return bytes(byte_array), padding_length
    
    def test_load_codebook_huffman_missing_padding(self):
        """Test loading Huffman codebook from file with missing padding length."""
        codebook = {0: '10', 1: '110', 2: '111'}
    
        # Write the Huffman codebook to the file, omitting the padding length for the last entry
        with open(self.test_file_path, "wb") as f:
            for i, (key, value) in enumerate(codebook.items()):
            # Write key length and key
                key_bytes = key.to_bytes((key.bit_length() + 7) // 8 or 1, byteorder='big')
                f.write(len(key_bytes).to_bytes(1, 'big'))
                f.write(key_bytes)
            
            # Convert value (bit string) to bytes
                value_bytes, padding_length = self.bits_to_bytes(value)
            
            # Write value length and value
                f.write(len(value_bytes).to_bytes(1, 'big'))
                f.write(value_bytes)
            
            # For the last entry, omit padding length to simulate a corrupted file
                if i < len(codebook) - 1:
                    f.write(padding_length.to_bytes(1, 'big'))

    # Try to load the Huffman codebook from the file
        with self.assertRaises(ValueError) as context:
            FileHandler.load_codebook_huffman(self.test_file_path)
    
    # Verify that the raised error is about missing padding length
        self.assertEqual(str(context.exception), "An error occurred while loading the codebook: Missing padding length")


    def test_load_codebook_huffman(self):
        """Test loading Huffman codebook from file."""
        codebook = {0: '10', 1: '110', 2: '111'}
 
        # Write the Huffman codebook to the file
        with open(self.test_file_path, "wb") as f:
            for key, value in codebook.items():
                # Write key length and key
                key_bytes = key.to_bytes((key.bit_length() + 7) // 8 or 1, byteorder='big')
                f.write(len(key_bytes).to_bytes(1, 'big'))
                f.write(key_bytes)
                
                # Convert value (bit string) to bytes
                value_bytes, padding_length = self.bits_to_bytes(value)
                
                # Write value length, value 
                f.write(len(value_bytes).to_bytes(1, 'big'))
                f.write(value_bytes)
                f.write(padding_length.to_bytes(1, 'big'))

        # Load the Huffman codebook from the file
        loaded_codebook = FileHandler.load_codebook_huffman(self.test_file_path)
        # Assert that the loaded codebook matches the expected codebook
        self.assertEqual(loaded_codebook, codebook)
        
        
    def test_load_codebook_huffman_incorrect_key_length(self):
        """Test loading Huffman codebook with incorrect key length."""
        with open(self.test_file_path, "wb") as f:
            f.write((2).to_bytes(1, 'big'))  # Key length = 2
            f.write(b'\x00')  # But only 1 byte for the key

        with self.assertRaises(ValueError):
            FileHandler.load_codebook_huffman(self.test_file_path)

    def test_load_codebook_huffman_incorrect_value_length(self):
        """Test loading Huffman codebook with incorrect value length."""
        with open(self.test_file_path, "wb") as f:
            f.write((1).to_bytes(1, 'big'))  # Key length = 1
            f.write(b'\x00')  # 1 byte key
            f.write((3).to_bytes(1, 'big'))  # Value length = 1
            f.write(b'\x00')  # 1 byte value
            
        with self.assertRaises(ValueError):
            FileHandler.load_codebook_huffman(self.test_file_path)


    def test_load_codebook_huffman_empty_file(self):
        """Test loading Huffman codebook from an empty file."""
        with open(self.test_file_path, 'wb') as f:
            pass

        loaded_codebook = FileHandler.load_codebook_huffman(self.test_file_path)
        self.assertEqual(loaded_codebook, {})            

    @patch('cv2.imread')
    @patch('src.compression_app.messagebox.showerror')
    def test_read_image_file_success(self, mock_showerror, mock_imread):
        """Test successful image file reading."""
        
        # Create dummy image data
        fake_image = np.array([[0, 0], [255, 255]], dtype=np.uint8)
        
        # Setting up a mock for cv2.imread
        mock_imread.return_value = fake_image
        
        #Calling a static method
        result = FileHandler.read_image_file('fake_image_path.png')
        
        # Check that the image is returned correctly
        self.assertTrue(np.array_equal(result, fake_image))
        mock_showerror.assert_not_called()  

    @patch('cv2.imread', return_value=None)
    @patch('src.compression_app.messagebox.showerror')
    def test_read_image_file_failure(self, mock_showerror, mock_imread):
        """Test failure in image file reading."""
        
        # ВCalling a static method
        result = FileHandler.read_image_file('fake_image_path.png')
        
        # Check that the result is None
        self.assertIsNone(result)
        
        # Check that the error message is shown
        mock_showerror.assert_called_once_with(
            "File read Error",
            "An error occurred while reading the file: The image file could not be loaded. Please check the file path and format."
        )

if __name__=='__main__':
    unittest.main()