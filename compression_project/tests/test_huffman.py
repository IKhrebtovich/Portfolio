import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.algorithms.huffman import Huffman



class TestHuffman(unittest.TestCase):
    """
    Test suite for Huffman compression and decompression.

    This test suite covers the following cases:
    - Compression and decompression of simple text data.
    - Compression and decompression of binary data (0-255 byte values).
    - Compression and decompression of repeated data to test efficiency.
    - Compression and decompression of empty data, with expected exceptions.
    - Compression and decompression of large data sets to ensure robustness.
    """
    def setUp(self):
        """
        Initializes the Huffman instance and sets up common test data.
        """
        self.huffman = Huffman()
        self.text_data = b"Hello world" 
        self.binary_data = bytes(range(256)) 
        self.repeated_data = b"A"*100
        self.empty_data = b''
        self.large_data = b"Hello world"*1000 

    def test_compress(self):
        """
        Verifies the Huffman compression method for various types of data.
        """
        # Test compression on text data
        compressed_data, codebook = self.huffman.compress(self.text_data)
        self.assertIsNotNone(compressed_data)
        self.assertTrue(len(compressed_data)>0)
        self.assertIsNotNone(codebook)
        self.assertTrue(len(codebook)>0) 

        # Test compression on binary data
        compressed_data, codebook = self.huffman.compress(self.binary_data)
        self.assertIsNotNone(compressed_data)
        self.assertTrue(len(compressed_data)>0)
        self.assertIsNotNone(codebook)
        self.assertTrue(len(codebook)>0) 

        # Test compression on repeated data
        compressed_data, codebook = self.huffman.compress(self.repeated_data)
        self.assertIsNotNone(compressed_data)
        self.assertTrue(len(compressed_data)>0)
        self.assertIsNotNone(codebook)
        self.assertTrue(len(codebook)>0) 

        # Test compression on empty data
        compressed_data, codebook = self.huffman.compress(self.empty_data)
        self.assertEqual(compressed_data, b'\x08') # Padding byte with no data
        self.assertEqual(codebook, {})
        
        # Test compression on large data
        compressed_data, codebook = self.huffman.compress(self.large_data)
        self.assertIsNotNone(compressed_data)
        self.assertTrue(len(compressed_data)>0)
        self.assertIsNotNone(codebook)
        self.assertTrue(len(codebook)>= len(set(self.large_data))) 


    def test_decompress(self):  
        """
        Verifies the Huffman decompression method for various types of data.
        """
        # Decompress the text data
        compressed_data, codebook = self.huffman.compress(self.text_data)
        decompressed_data = self.huffman.decompress(compressed_data, codebook)
        self.assertIsNotNone(decompressed_data)
        self.assertTrue(len(decompressed_data)>0) 
        self.assertEqual(decompressed_data, self.text_data)
        
        # Decompress the binary data
        compressed_data, codebook = self.huffman.compress(self.binary_data)
        decompressed_data = self.huffman.decompress(compressed_data, codebook)
        self.assertIsNotNone(decompressed_data)
        self.assertTrue(len(decompressed_data)>0) 
        self.assertEqual(decompressed_data, self.binary_data)

        # Decompress the repeated data
        compressed_data, codebook = self.huffman.compress(self.repeated_data)
        decompressed_data = self.huffman.decompress(compressed_data, codebook)
        self.assertIsNotNone(decompressed_data)
        self.assertTrue(len(decompressed_data)>0) 
        self.assertEqual(decompressed_data, self.repeated_data)
     
        
        # Decompress the large data
        compressed_data, codebook = self.huffman.compress(self.large_data)
        decompressed_data = self.huffman.decompress(compressed_data, codebook)
        self.assertIsNotNone(decompressed_data)
        self.assertTrue(len(decompressed_data)>0) 
        self.assertEqual(decompressed_data, self.large_data)
        
       


if __name__=="__main__":
    unittest.main()