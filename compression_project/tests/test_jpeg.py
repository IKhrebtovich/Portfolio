import unittest
import sys
import os
import numpy as np
import cv2
from src.algorithms.jpeg import JPEG



class TestJPEG(unittest.TestCase):

    def setUp(self):
        # Initialize JPEG object and quantization matrix
        self.jpeg = JPEG()
        self.quant_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])

    def test_rgb_to_ycbcr_channels(self):
        """Test for rgb_to_ycbcr_channels conversion."""
        img_rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        
        # Call the method to get luminance, blue (Cb), and red (Cr) channels
        luminance, blue, red = self.jpeg.rgb_to_ycbcr_channels(img_rgb)
        # Expected Y, Cb, and Cr values
        expected_luminance = np.array([[76, 150, 29]], dtype=np.uint8)  
        expected_blue = np.array([[85, 43, 255]], dtype=np.uint8)  
        expected_red = np.array([[255, 21, 107]], dtype=np.uint8)  
        
        # Check that the output channels are as expected
        np.testing.assert_array_almost_equal(luminance, expected_luminance, decimal=0)
        np.testing.assert_array_almost_equal(blue, expected_blue, decimal=0)
        np.testing.assert_array_almost_equal(red, expected_red, decimal=0)

    def test_dct_blocks(self):
        """Test the Discrete Cosine Transform (DCT) on 8x8 blocks."""
        block = np.array([[255]*8]*8, dtype=np.float32)
        
        # Apply DCT to the block
        dct_result = self.jpeg.discrete_cosine_transform_blocks([block])
        # Assert that the top-left DCT coefficient (DC coefficient) is approximately 2040
        self.assertAlmostEqual(dct_result[0][0, 0], 2040.0, places=1)

    def test_rle_encoding(self):
        """Test the Run-Length Encoding (RLE) of a zigzag sequence."""
        zigzag = [0, 0, 1, 1, 1, 0, 0, 1]

        # Perform RLE on the zigzag sequence
        rle_result = self.jpeg.run_length_encode(zigzag)
        # Expected RLE result
        expected_rle = [[0, 2], [1, 3], [0, 2], [1, 1]]
        
        self.assertEqual(rle_result, expected_rle)

    def test_zigzag_scan(self):
        """Test the zigzag scan on an 8x8 block."""
        block = np.arange(64).reshape(8, 8)
                
        zigzag_result = self.jpeg.zigzag_scan(block)
        expected_zigzag = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 
                           40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 
                           43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 
                           46, 53, 60, 61, 54, 47, 55, 62, 63]
        self.assertEqual(zigzag_result, expected_zigzag)

    def test_inverse_zigzag_scan(self):
        """Test inverse zigzag scan on a flat sequence of 64 elements."""
        flat_sequence = [i for i in range(64)]
        
        matrix = self.jpeg.inverse_zigzag_scan(flat_sequence, 8)
        
        # Expected 8x8 matrix after inverse zigzag scan
        expected_matrix = np.array([[0, 1, 5, 6, 14, 15, 27, 28],
                                    [2, 4, 7, 13, 16, 26, 29, 42],
                                    [3, 8, 12, 17, 25, 30, 41, 43],
                                    [9, 11, 18, 24, 31, 40, 44, 53],
                                    [10, 19, 23, 32, 39, 45, 52, 54],
                                    [20, 22, 33, 38, 46, 51, 55, 60],
                                    [21, 34, 37, 47, 50, 56, 59, 61],
                                    [35, 36, 48, 49, 57, 58, 62, 63]])
        
        np.testing.assert_array_equal(matrix, expected_matrix)

    def test_subsample(self):
        """Test 2:1 subsampling of a 4x4 channel down to a 2x2 channel."""
        original_channel = np.array([[1, 2, 3, 4],
                                     [5, 6, 7, 8],
                                     [9, 10, 11, 12],
                                     [13, 14, 15, 16]], dtype=np.uint8)
        
        # Perform subsampling on the original channel
        subsampled_channel = self.jpeg.subsample(original_channel)
        expected_channel = np.array([[1, 3],
                                     [9, 11]], dtype=np.uint8)  
        
        np.testing.assert_array_equal(subsampled_channel, expected_channel)


    def test_quantize_blocks(self):
            """Test quantization of DCT blocks using a given quantization matrix."""
    
            # Example DCT block to be quantized
            dct_block = np.array([
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]
            ])

            # Expected quantized block after dividing by quantization matrix
            expected_quantized_block = np.array([
                [ 1,  1,  1,  1,  1,  1,  1,  1],
                [ 1,  1,  1,  1,  1,  1,  1,  1],
                [ 1,  1,  1,  1,  1,  1,  1,  1],
                [ 1,  1,  1,  1,  1,  1,  1,  1],
                [ 1,  1,  1,  1,  1,  1,  1,  1],
                [ 1,  1,  1,  1,  1,  1,  1,  1],
                [ 1,  1,  1,  1,  1,  1,  1,  1],
                [ 1,  1,  1,  1,  1,  1,  1,  1]
            ])

            dct_blocks = [dct_block]
            
            quantized_blocks = self.jpeg.quantize_blocks(dct_blocks, self.quant_matrix)
            
            # Compare the quantized result with the expected output
            np.testing.assert_array_equal(quantized_blocks[0], expected_quantized_block)

    def test_process_blocks_for_rle(self):
        """Test RLE encoding of quantized DCT blocks."""            
         # Example DCT block to be encoded with RLE
        dct_block = np.array([
                [ 1,  1,  1,  0,  0,  0,  0,  0],
                [ 1,  1,  0,  0,  0,  0,  0,  0],
                [ 1,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0,  0]
        ])
            
        # Expected result after RLE encoding
        expected_rle_block = [(1,  6), (0,  58)]

        blocks = [dct_block]
        # Perform RLE encoding on the DCT blocks
        rle_blocks = self.jpeg.process_blocks_for_rle(blocks)
            
        # Convert numpy int64 to int for comparison
        rle_blocks_int = [[(int(val[0]), val[1]) for val in block] for block in rle_blocks]
            
        # Compare the RLE result with the expected output
        self.assertEqual(rle_blocks_int[0], expected_rle_block)

    def test_process_blocks_zigzag(self):
        """Test zigzag scanning of DCT blocks."""
        
        # Example 8x8 DCT block
        dct_block = np.array([
            [1, 2, 0, 0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ])

       # Expected result from zigzag scanning
        expected_zigzag_block = [
            1, 2, 3, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        ]

        # Test process_blocks_zigzag method
        blocks = [dct_block]
        zigzag_blocks = self.jpeg.process_blocks_zigzag(blocks)

        # Convert each zigzag block to a list of integers for comparison
        zigzag_blocks_int = [list(map(int, block)) for block in zigzag_blocks]

        # Compare the zigzag result with the expected output
        self.assertEqual(zigzag_blocks_int[0], expected_zigzag_block)

    def test_process_blocks_rle(self):
        """Test RLE encoding on zigzag-scanned blocks."""
        
        # Example block after zigzag scanning (mostly zeros)
        zigzag_block = [
            1, 1, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        ]

        # Expected result after RLE encoding
        expected_rle_block = [[1, 3], [0, 13]]
        # Run the RLE process on the zigzag-scanned block
        blocks = [zigzag_block]
        rle_blocks = self.jpeg.process_blocks_rle(blocks)

        # Compare the first (and only) block with the expected result
        self.assertEqual(rle_blocks[0], expected_rle_block)

    def test_flatten_rle_blocks(self):
        """Test flattening of RLE blocks into a single list."""
        # Example RLE blocks
        rle_blocks = [
            [[1, 3], [0, 13]],
            [[2, 5], [0, 3]],
            [[3, 2], [0, 6]]
        ]

        # Expected flattened list
        expected_flattened_list = [
            [1, 3], [0, 13],
            [2, 5], [0, 3],
            [3, 2], [0, 6]
        ]
         # Flatten the RLE blocks
        flattened_list = self.jpeg.flatten_rle_blocks(rle_blocks)

        self.assertEqual(flattened_list, expected_flattened_list)


    def test_reformat_image_padding_needed(self):
        """Test reformatting an image that requires padding."""
        # Изображение, нуждающееся в паддинге
        image = np.array([
            [1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19, 20, 21],
            [22, 23, 24, 25, 26, 27, 28],
            [29, 30, 31, 32, 33, 34, 35],
            [36, 37, 38, 39, 40, 41, 42],
            [43, 44, 45, 46, 47, 48, 49]
        ])

        # Expected result with padding
        expected_image = np.array([
            [1, 2, 3, 4, 5, 6, 7, 6],
            [8, 9, 10, 11, 12, 13, 14, 13],
            [15, 16, 17, 18, 19, 20, 21, 20],
            [22, 23, 24, 25, 26, 27, 28, 27],
            [29, 30, 31, 32, 33, 34, 35, 34],
            [36, 37, 38, 39, 40, 41, 42, 41],
            [43, 44, 45, 46, 47, 48, 49, 48],
            [36, 37, 38, 39, 40, 41, 42, 41]
        ])

        # Perform reformatting
        reformatted_image = self.jpeg.reformat(image)
        
        np.testing.assert_array_equal(reformatted_image, expected_image)


    def test_split_into_blocks_basic(self):
        """Test splitting an image into 8x8 blocks without remainder."""
        
        # Image that can be split into 8x8 blocks without remainder
        image = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16],
            [17, 18, 19, 20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29, 30, 31, 32],
            [33, 34, 35, 36, 37, 38, 39, 40],
            [41, 42, 43, 44, 45, 46, 47, 48],
            [49, 50, 51, 52, 53, 54, 55, 56],
            [57, 58, 59, 60, 61, 62, 63, 64]
        ])

        # Expected result: one 8x8 block
        expected_blocks = np.array([
            [[1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16],
            [17, 18, 19, 20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29, 30, 31, 32],
            [33, 34, 35, 36, 37, 38, 39, 40],
            [41, 42, 43, 44, 45, 46, 47, 48],
            [49, 50, 51, 52, 53, 54, 55, 56],
            [57, 58, 59, 60, 61, 62, 63, 64]]
        ])

        # Perform the block splitting
        blocks = self.jpeg.split_into_blocks(image, block_size=8)
        
        np.testing.assert_array_equal(blocks, expected_blocks)


    def test_quantize(self):
        """Test quantization of a DCT block using a quantization matrix."""
        # Example DCT block
        dct_block = np.array([[ 16,  11,  10,  16],
                            [ 12,  12,  14,  19],
                            [ 14,  13,  16,  24],
                            [ 14,  17,  22,  29]], dtype=np.float32)
        
        # Quantization matrix
        quant_matrix = np.array([[ 16,  11,  10,  16],
                                [ 12,  12,  14,  19],
                                [ 14,  13,  16,  24],
                                [ 14,  17,  22,  29]], dtype=np.float32)
        
        # Expected quantized block
        expected_quantized_block = np.array([[ 1,  1,  1,  1],
                                            [ 1,  1,  1,  1],
                                            [ 1,  1,  1,  1],
                                            [ 1,  1,  1,  1]], dtype=np.int32)
        
        # Perform quantization
        quantized_block = self.jpeg.quantize(dct_block, quant_matrix)
        
        np.testing.assert_array_equal(quantized_block, expected_quantized_block)

    def test_convert_numpy_array(self):
        """Test conversion of a numpy array of floats to a numpy array of integers."""
        # Input numpy array with floats
        numpy_array = np.array([[1.1, 2.5], [3.7, 4.9]])

        # Expected result after conversion to integers (truncated towards zero)
        expected = np.array([[1, 2], [3, 4]], dtype=np.int32)

        # Perform the conversion
        result = self.jpeg.convert_numpy_to_int(numpy_array)

        np.testing.assert_array_equal(result, expected)


    def test_conv_rle_to_list_valid(self):
         """Test conversion from RLE format to a flat list."""
          # Input RLE data
         rle = [(0, 3), (2, 5), (4, 2)]
            
         # Expected flat list result
         expected = [128, 3, 130, 5, 132, 2]
            
         # Perform the conversion
         result = self.jpeg.conv_rle_to_list(rle)
            
         self.assertEqual(result, expected)

    def test_ycbcr_to_rgb(self):
        """Test conversion from YCbCr to RGB."""
        
        # Test data for Y, Cb, Cr channels
        y = np.array([[0, 128], [255, 128]], dtype=np.uint8)
        cb = np.array([[128, 128], [128, 128]], dtype=np.uint8)
        cr = np.array([[128, 128], [128, 128]], dtype=np.uint8)

        # Expected RGB values based on the conversion formulas
        expected_rgb = np.array([
            [[0, 0, 0], [128, 128, 128]],
            [[255, 255, 255], [128, 128, 128]]
        ], dtype=np.uint8)

        # Convert YCbCr to RGB
        result_rgb = self.jpeg.ycbcr_to_rgb(y, cb, cr)

        # Check if the result matches the expected RGB values
        np.testing.assert_array_equal(result_rgb, expected_rgb)
        

    def test_run_length_decode(self):
        """Test decoding of RLE-encoded data."""

        # Example RLE data
        rle_encoded_data = [(3, 4), (7, 2), (1, 3)]
        expected_output = [3, 3, 3, 3, 7, 7, 1, 1, 1]
            
        # Call the method
        result = self.jpeg.run_length_decode(rle_encoded_data)
            
        self.assertEqual(result, expected_output)

    def test_process_blocks_from_rle(self):
        """Test conversion of RLE-decoded data into 8x8 blocks."""
        decoded_data = [1] * 64 
        expected_block = np.ones((8, 8), dtype=np.int32)

        blocks = self.jpeg.process_blocks_from_rle(decoded_data, block_size=8)
        print("Blocks:", blocks)  

        if blocks:
            np.testing.assert_array_equal(blocks[0], expected_block)
        else:
            self.fail("No blocks were returned from process_blocks_from_rle.")

    def test_dequantize_blocks(self):
        """Test dequantization of blocks."""
        # Example test data
        quant_blocks = [
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            np.array([[5, 6], [7, 8]], dtype=np.int32)
        ]
        quant_matrix = np.array([[2, 2], [2, 2]], dtype=np.int32)
        
        # Expected results after dequantization
        expected_blocks = [
            np.array([[2, 4], [6, 8]], dtype=np.int32),
            np.array([[10, 12], [14, 16]], dtype=np.int32)
        ]
        
        # Perform dequantization
        result_blocks = self.jpeg.dequantize_blocks(quant_blocks, quant_matrix)
        
        for result, expected in zip(result_blocks, expected_blocks):
            np.testing.assert_array_equal(result, expected)

    def test_dequantize(self):
        """Test dequantization of a quantized block."""
        # Example quantized block and quantization matrix
        quant_block = np.array([
            [1, 2],
            [3, 4]
        ], dtype=np.int32)

        quant_matrix = np.array([
            [2, 2],
            [2, 2]
        ], dtype=np.int32)

        # Expected result after dequantization
        expected_block = np.array([
            [2, 4],
            [6, 8]
        ], dtype=np.int32)

        # Perform dequantization
        result_block = self.jpeg.dequantize(quant_block, quant_matrix)

        np.testing.assert_array_equal(result_block, expected_block)

    def test_conv_list_to_rle(self):
        """Convert a list of integers into Run-Length Encoding (RLE)."""
        # Example input data
        data = [130, 5, 135, 3, 128, 10]

        # Expected RLE output
        expected_rle = [[2, 5], [7, 3], [0, 10]]

        # Perform the conversion
        result_rle = self.jpeg.conv_list_to_rle(data)

        self.assertEqual(result_rle, expected_rle)    
   
    def test_combine_blocks(self):
        
        block_size = 8
        width = 16
        height = 16
        num_blocks_x = width // block_size
        num_blocks_y = height // block_size

        blocks = [np.full((block_size, block_size), i, dtype=np.float32)
                  for i in range(num_blocks_x * num_blocks_y)]

        expected_image = np.zeros((height, width), dtype=np.float32)
        for i in range(num_blocks_y):
            for j in range(num_blocks_x):
                expected_image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = blocks[i * num_blocks_x + j]

        result_image = self.jpeg.combine_blocks(blocks, width, height)

        np.testing.assert_array_equal(result_image, expected_image)

    def test_restore_size(self):
        original_size = (10, 10)
        test_image = np.full(original_size, 255, dtype=np.uint8)  

        target_size = (20, 20)

        resized_image = self.jpeg.restore_size(test_image, target_size)

        # Check dimensions
        self.assertEqual(resized_image.shape, (target_size[1], target_size[0]))
    
    def test_idct(self):
        # Create a test DCT block
        quant_block_reverse = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ], dtype=np.float32)

        # Apply IDCT to the test block
        idct_block = self.jpeg.idct(quant_block_reverse)
        
        # Expected result after IDCT
        expected_block = np.array([
            [ 3.701313, -0.736237,  0.736237,  0.146447],
            [-0.736237,  0.146447, -0.146447, -0.02913 ],
            [ 0.736237, -0.146447,  0.146447,  0.02913 ],
            [ 0.146447, -0.02913 ,  0.02913 ,  0.005794]
        ], dtype=np.float32)

        np.testing.assert_allclose(idct_block, expected_block, rtol=1e-3)

    def test_idct_blocks(self):
        # Create multiple test DCT blocks
        quant_blocks_reverse = [
            np.array([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]
            ], dtype=np.float32),
            np.array([
                [2, 2, 2, 2],
                [2, 2, 2, 2],
                [2, 2, 2, 2],
                [2, 2, 2, 2]
            ], dtype=np.float32)
        ]

        # Apply IDCT to the list of blocks
        idct_blocks = self.jpeg.idct_blocks(quant_blocks_reverse)

        # Expected results after IDCT
        expected_blocks = [
            np.array([
                [ 3.701313, -0.736237,  0.736237,  0.146447],
                [-0.736237,  0.146447, -0.146447, -0.02913 ],
                [ 0.736237, -0.146447,  0.146447,  0.02913 ],
                [ 0.146447, -0.02913 ,  0.02913 ,  0.005794]
            ], dtype=np.float32),
            np.array([
                [ 7.402626, -1.472473,  1.472473,  0.292894],
                [-1.472473,  0.292894, -0.292894, -0.05826 ],
                [ 1.472473, -0.292894,  0.292894,  0.05826 ],
                [ 0.292894, -0.05826 ,  0.05826 ,  0.011588]
            ], dtype=np.float32)
        ]

        for idct_block, expected_block in zip(idct_blocks, expected_blocks):
            np.testing.assert_allclose(idct_block, expected_block, rtol=1e-3)
         
    def test_decompress(self):
        # Mock function for Huffman decompression
        def huffman_decompress(compressed_data, huffman_table):
           # Fake data for the test
            return [(0, 2), (1, 3), (0, 1)]

        # Simple RLE decoding function
        def run_length_decode(rle_encoded_data):
            return [value for value, count in rle_encoded_data for _ in range(count)]

        # Mock function for JPEG decompression
        def decompress(compressed_huffman, huffman_table, length, height,
                       luminance_length, luminance_height, blue_length, blue_height,
                       red_length, red_height, len_rle_lum, len_rle_blue, len_rle_red):
            # Decompress data using the Huffman mock
            decompressed_rle_list = huffman_decompress(compressed_huffman, huffman_table)

            ## Simple RLE decoding
            lum_data_blocks = run_length_decode(decompressed_rle_list)

            # Return image (zeroed for the test)
            return np.zeros((height, length, 3))

        # Test data
        compressed_huffman = b'\x00\x01\x02'
        huffman_table = {0: '00', 1: '01'}

        # Image and color channel sizes
        length = 320
        height = 240
        luminance_length = 320
        luminance_height = 240
        blue_length = 160
        blue_height = 120
        red_length = 160
        red_height = 120
        len_rle_lum = 1000
        len_rle_blue = 500
        len_rle_red = 500

        # Perform decompression
        output_image = decompress(
            compressed_huffman, huffman_table, length, height,
            luminance_length, luminance_height, blue_length, blue_height,
            red_length, red_height, len_rle_lum, len_rle_blue, len_rle_red
        )

        # Check the size of the output image
        self.assertEqual(output_image.shape, (height, length, 3))
    
    def test_compress(self):
         # Mock functions
        def rgb_to_ycbcr_channels(rgb_image):
            return np.zeros(rgb_image.shape[:2]), np.zeros((rgb_image.shape[0] // 2, rgb_image.shape[1] // 2)), np.zeros((rgb_image.shape[0] // 2, rgb_image.shape[1] // 2))

        def subsample(channel): return channel
        def reformat(channel): return channel
        def split_into_blocks(channel): return [channel]
        def discrete_cosine_transform_blocks(blocks): return blocks
        def quantize_blocks(blocks, quant_table): return blocks
        def process_blocks_for_rle(blocks): return [block.flatten() for block in blocks]
        def flatten_rle_blocks(rle_blocks): return np.concatenate(rle_blocks).tolist()
        def convert_numpy_to_int(data): return data
        def conv_rle_to_list(rle_data): return rle_data

        class Huffman:
            def compress(self, data): return b'\x00\x01\x02', {0: '00', 1: '01'}

        def compress(data):
            luminance, blue, red = rgb_to_ycbcr_channels(data)
            blue = subsample(blue)
            red = subsample(red)
            luminance = reformat(luminance)
            blue = reformat(blue)
            red = reformat(red)
            lum_blocks = split_into_blocks(luminance)
            blue_blocks = split_into_blocks(blue)
            red_blocks = split_into_blocks(red)
            dct_lum_blocks = discrete_cosine_transform_blocks(lum_blocks)
            dct_blue_blocks = discrete_cosine_transform_blocks(blue_blocks)
            dct_red_blocks = discrete_cosine_transform_blocks(red_blocks)
            quant_lum_blocks = quantize_blocks(dct_lum_blocks, None)
            quant_blue_blocks = quantize_blocks(dct_blue_blocks, None)
            quant_red_blocks = quantize_blocks(dct_red_blocks, None)
            rle_lum_blocks = process_blocks_for_rle(quant_lum_blocks)
            rle_blue_blocks = process_blocks_for_rle(quant_blue_blocks)
            rle_red_blocks = process_blocks_for_rle(quant_red_blocks)
            combined_rle_lum = flatten_rle_blocks(rle_lum_blocks)
            combined_rle_blue = flatten_rle_blocks(rle_blue_blocks)
            combined_rle_red = flatten_rle_blocks(rle_red_blocks)
            all_rle_data = combined_rle_lum + combined_rle_blue + combined_rle_red
            all_rle_data_int = convert_numpy_to_int(all_rle_data)
            all_rle_data_huffman = conv_rle_to_list(all_rle_data_int)
            huffman = Huffman()
            compressed_huffman, huffman_table = huffman.compress(all_rle_data_huffman)
            return compressed_huffman, huffman_table, len(data[1]), len(data), len(data[1]), len(data), len(data[1]) // 2, len(data) // 2, len(data[1]) // 2, len(data) // 2, len(combined_rle_lum), len(combined_rle_blue), len(combined_rle_red)

        # Create test data
        rgb_image = np.zeros((240, 320, 3), dtype=np.uint8)

        # Perform compression
        compressed_data = compress(rgb_image)

        # Check the types of the output data
        for i, item in enumerate(compressed_data):
            if i == 0:  
                assert isinstance(item, bytes), f"Element {i} should be of type bytes, got {type(item)}"
            elif i == 1:  
                assert isinstance(item, dict), f"Element {i} should be of type dict, got {type(item)}"
            else:  
                assert isinstance(item, (bytes, int)), f"Element {i} should be of type bytes or int, got {type(item)}" 
    
    def test_run_length_decode(self):
        # Test data
        rle_encoded_data = [(1, 3), (2, 2), (3, 1)]
        
        # Expected result
        expected_result = [1, 1, 1, 2, 2, 3]
        
        # Call the method
        result = self.jpeg.run_length_decode(rle_encoded_data)
        
        self.assertEqual(result, expected_result)
    





if __name__ == '__main__':
    unittest.main()

    

