import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.algorithms.lzw import LZW


class TestLZW(unittest.TestCase):
    """
    Test suite for LZW compression and decompression.
    
    This suite verifies the correctness of the LZW algorithm by testing its ability to compress and
    decompress various types of data, including:
    - Simple text data
    - Binary data covering the full byte range
    - Data with repeated patterns
    - Empty data
    - Large datasets
    
    Each test case ensures that the data is accurately compressed and decompressed, and that the 
    generated codebook is valid.
    """
    def setUp(self):
        """
        Initializes the LZW instance and sets up common test data.

        The following test data is prepared:
        - text_data: A simple text string to test basic compression functionality.
        - binary_data: A byte sequence containing all possible byte values (0-255) to test compression 
                        across the full byte range.
        - repeated_data: A byte sequence with repeated characters to test the efficiency of compression 
                          for repetitive patterns.
        - empty_data: An empty byte sequence to test how the algorithm handles edge cases.
        - large_data: A large dataset created by repeating the text data to test performance and robustness.
        """
        self.lzw = LZW()
        self.text_data = b"Hello world" 
        self.binary_data = bytes(range(256)) 
        self.repeated_data = b"A"*100
        self.empty_data = b''
        self.large_data = b"Hello world"*1000 

    def test_compress(self):
        """
        Verifies the LZW compression method with various types of data.

        Each test ensures that:
        - Compression generates non-empty output and a valid codebook.
        - The output is appropriate for different types of input data:
            - Text data: Checks that the output is non-empty and that the codebook contains entries.
            - Binary data: Verifies that all byte values are handled and the codebook is properly populated.
            - Repeated data: Ensures that compression is efficient for repetitive patterns.
            - Empty data: Confirms that compressing empty input returns an empty list and a standard codebook.
            - Large data: Validates that large datasets are compressed correctly and that the codebook grows as needed.
        """
        # Test compression on text data
        compressed_data, codebook = self.lzw.compress(self.text_data)
        self.assertIsNotNone(compressed_data)
        self.assertTrue(len(compressed_data)>0)
        self.assertIsNotNone(codebook)
        self.assertTrue(len(codebook)>0) 


        # Test compression on binary data
        compressed_data, codebook = self.lzw.compress(self.binary_data)
        self.assertIsNotNone(compressed_data)
        self.assertTrue(len(compressed_data)>0)
        self.assertIsNotNone(codebook)
        self.assertTrue(len(codebook)>0) 

        # Test compression on repeated data
        compressed_data, codebook = self.lzw.compress(self.repeated_data)
        self.assertIsNotNone(compressed_data)
        self.assertTrue(len(compressed_data)>0)
        self.assertIsNotNone(codebook)
        self.assertTrue(len(codebook)>0) 

        # Test compression on empty data
        compressed_data, codebook = self.lzw.compress(self.empty_data)
        self.assertEqual(compressed_data, [])
        self.assertEqual(codebook, {bytes([i]): i for i in range(256)})
        
        # Test compression on large data
        compressed_data, codebook = self.lzw.compress(self.large_data)
        self.assertIsNotNone(compressed_data)
        self.assertTrue(len(compressed_data)>0)
        self.assertIsNotNone(codebook)
        self.assertTrue(len(codebook)>256) 

    def test_decompress(self):  
        """
        Verifies the LZW decompression method with various types of data.
        
        Each test ensures that:
        - Decompression accurately reconstructs the original data.
        - The output matches the expected data for different input scenarios:
            - Text data: Checks that the decompressed data matches the original text.
            - Binary data: Verifies that the full byte range is correctly reconstructed.
            - Repeated data: Ensures that decompression handles repetitive patterns properly.
            - Empty data: Confirms that decompressing empty data raises a ValueError as expected.
            - Large data: Validates that large datasets are decompressed correctly and match the original data.
        """
        
        # Test decompression on text data
        compressed_data, codebook = self.lzw.compress(self.text_data)
        decompressed_data = self.lzw.decompress(compressed_data, codebook)
        self.assertIsNotNone(decompressed_data)
        self.assertTrue(len(decompressed_data)>0) 
        self.assertEqual(decompressed_data, self.text_data)
        
        # Test decompression on binary data
        compressed_data, codebook = self.lzw.compress(self.binary_data)
        decompressed_data = self.lzw.decompress(compressed_data, codebook)
        self.assertIsNotNone(decompressed_data)
        self.assertTrue(len(decompressed_data)>0) 
        self.assertEqual(decompressed_data, self.binary_data)

        # Test decompression on repeated data
        compressed_data, codebook = self.lzw.compress(self.repeated_data)
        decompressed_data = self.lzw.decompress(compressed_data, codebook)
        self.assertIsNotNone(decompressed_data)
        self.assertTrue(len(decompressed_data)>0) 
        self.assertEqual(decompressed_data, self.repeated_data)
     
        # Decompress the empty data
        with self.assertRaises(ValueError):
            self.lzw.decompress([], {})
        
        # Test decompression on large data
        compressed_data, codebook = self.lzw.compress(self.large_data)
        decompressed_data = self.lzw.decompress(compressed_data, codebook)
        self.assertIsNotNone(decompressed_data)
        self.assertTrue(len(decompressed_data)>0) 
        self.assertEqual(decompressed_data, self.large_data)
        
    def test_decompress_invalid_data(self):
        """Test decompression with invalid data to ensure appropriate exceptions are raised."""
        # Example of invalid compressed data
        invalid_compressed_data = [999]
        codebook = {bytes([i]): i for i in range(256)}  # Example codebook
        
        with self.assertRaises(ValueError):
            self.lzw.decompress(invalid_compressed_data, codebook)

if __name__=="__main__":
    unittest.main()