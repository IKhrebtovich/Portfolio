import numpy as np
import cv2
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from huffman import Huffman

class JPEG:
    """
    A class for JPEG image processing.
    Handles RGB to YCbCr color space conversion, subsampling, block splitting,
    Discrete Cosine Transform (DCT), quantization, and encoding using RLE and Huffman coding.
    """
    
    def __init__(self):
        """
        Initializes parameters for image conversion and quantization.
        """
        # Matrix for converting RGB to YCbCr color space
        self.con_mat = np.array(
            [[0.299, 0.587, 0.114], 
             [-0.1687, -0.3313, 0.500], 
             [0.500, -0.4187, -0.0813]]
        )
        # Offset added during conversion to YCbCr
        self.sub_mat = np.array([0, 128, 128])
        
        # Inverse matrix for converting YCbCr back to RGB
        self.inv_con_mat = np.linalg.inv(self.con_mat)
        
        # Offset added during conversion back to RGB
        self.add_mat = np.array([0, -128, -128])
        
       # Quantization matrices used to quantize DCT coefficients
        self.quant_lum = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])

        self.quant_colour = np.array([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ]) 
        

    def compress(self, data):
        """
        Сompresses an RGB image using JPEG compression techniques.

        Args:
            data (numpy array): RGB image data.

        Returns:
            tuple: Contains compressed Huffman data, Huffman table, and metadata.
        """
        # Convert RGB image to YCbCr color space and extract channels
        img_rgb = data
        luminance, blue, red = self.rgb_to_ycbcr_channels(img_rgb)
        

        # Apply subsampling (4:2:0) to Cb and Cr channels
        blue = self.subsample(blue)
        red = self.subsample(red)
        
        # Pad channels to be multiples of 8
        luminance = self.reformat(luminance)
        blue = self.reformat(blue)
        red = self.reformat(red)
        
        # Split channels into 8x8 blocks
        lum_blocks = self.split_into_blocks(luminance)
        blue_blocks = self.split_into_blocks(blue)
        red_blocks = self.split_into_blocks(red)
        
         # Apply Discrete Cosine Transform (DCT)
        dct_lum_blocks = self.discrete_cosine_transform_blocks(lum_blocks)
        dct_blue_blocks = self.discrete_cosine_transform_blocks(blue_blocks)
        dct_red_blocks = self.discrete_cosine_transform_blocks(red_blocks)
        
        # Apply quantize DCT blocks
        quant_lum_blocks = self.quantize_blocks(dct_lum_blocks, self.quant_lum)
        quant_blue_blocks = self.quantize_blocks(dct_blue_blocks, self.quant_colour)
        quant_red_blocks = self.quantize_blocks(dct_red_blocks, self.quant_colour)

        # Encode blocks using Run-Length Encoding (RLE)
        rle_lum_blocks = self.process_blocks_for_rle(quant_lum_blocks)
        rle_blue_blocks = self.process_blocks_for_rle(quant_blue_blocks)
        rle_red_blocks = self.process_blocks_for_rle(quant_red_blocks)
               
        # Flatten RLE-encoded blocks 
        combined_rle_lum = self.flatten_rle_blocks(rle_lum_blocks)
        combined_rle_blue = self.flatten_rle_blocks(rle_blue_blocks)
        combined_rle_red = self.flatten_rle_blocks(rle_red_blocks)
        
        # Combine all RLE-encoded data for Huffman encoding
        all_rle_data = combined_rle_lum + combined_rle_blue + combined_rle_red
        all_rle_data_int = self.convert_numpy_to_int(all_rle_data)
        all_rle_data_huffman = self.conv_rle_to_list(all_rle_data_int)
                
        # Compress data using Huffman coding
        huffman = Huffman()
        compressed_huffman, huffman_table = huffman.compress(all_rle_data_huffman)
        
        # Calculate image dimensions and RLE data lengths
        height, width = img_rgb.shape[:2]
        luminance_height, luminance_width = luminance.shape
        blue_height, blue_width = blue.shape
        red_height, red_width = red.shape
        
        # Convert dimensions and lengths to bytes
        length = width.to_bytes(3, byteorder='big') 
        height = height.to_bytes(3, byteorder='big') 
        luminance_length = luminance_width.to_bytes(3, byteorder='big')
        luminance_height = luminance_height.to_bytes(3, byteorder='big')
        blue_length = blue_width.to_bytes(3, byteorder='big')  
        blue_height = blue_height.to_bytes(3, byteorder='big')  
        red_length = red_width.to_bytes(3, byteorder='big')  
        red_height = red_height.to_bytes(3, byteorder='big')  
        len_rle_lum = len(combined_rle_lum)
        len_rle_blue =len(combined_rle_blue)
        len_rle_red = len(combined_rle_red)
        
        len_rle_lum = len_rle_lum.to_bytes(3, byteorder='big')
        len_rle_blue =len_rle_blue.to_bytes(3, byteorder='big')
        len_rle_red = len_rle_red.to_bytes(3, byteorder='big')

        
        return compressed_huffman, huffman_table, length, height, luminance_length, luminance_height, blue_length, blue_height, red_length, red_height, len_rle_lum, len_rle_blue, len_rle_red

    def rgb_to_ycbcr_channels(self, img_rgb):
        """
        Converts an RGB image to YCbCr color space and extracts the Y, Cb, and Cr channels.

        Args:
            img_rgb (numpy array): Input RGB image.
        Returns:
            tuple of three numpy arrays representing the Y, Cb, and Cr channels respectively:
        """
        luminance = np.zeros_like(img_rgb[:, :, 0])
        blue = np.zeros_like(img_rgb[:, :, 0])
        red = np.zeros_like(img_rgb[:, :, 0])

        # Convert each RGB pixel to YCbCr 
        for row in range(img_rgb.shape[0]):
            for col in range(img_rgb.shape[1]):
                YCbCr_pixel = self.convert_to_YCbCr(img_rgb[row, col])
                luminance[row, col] = YCbCr_pixel[0]
                blue[row, col] = YCbCr_pixel[1]
                red[row, col] = YCbCr_pixel[2]
        return luminance, blue, red
    
    def convert_to_YCbCr(self, rgb):
        """
        Converts an RGB color value to YCbCr color space.

        Args:
            rgb (numpy.ndarray): A 1D array or list with RGB values [R, G, B].

        Returns:
            numpy.ndarray: A 1D array with YCbCr values [Y, Cb, Cr].
        """
        return np.dot(rgb, self.con_mat.T) + self.sub_mat

    def discrete_cosine_transform_blocks(self, blocks):
        """
        Applies the Discrete Cosine Transform (DCT) to a list of image blocks.

        This method performs the DCT on each block of the image and returns the transformed blocks. 
        It is intended for use in JPEG compression, where each 8x8 block of the image is transformed 
        individually.

        Args:
            blocks (list of numpy.ndarray): A list of 2D numpy arrays representing image blocks. 
            Each block should have a shape of (8, 8) and contain integer pixel values.

        Returns:
            list of numpy.ndarray: A list of 2D numpy arrays, where each block contains DCT-transformed 
            coefficients with `float32` data type. These coefficients represent frequency components of the image block.
        """
        return [cv2.dct(block.astype(np.float32)) for block in blocks]

    def quantize_blocks(self, dct_blocks, quant_matrix):
        """
        Quantizes a list of DCT-transformed image blocks.

        This method applies quantization to each block of DCT coefficients using a predefined quantization matrix.
        Quantization reduces the precision of the DCT coefficients to compress the image further.

        Args:
            dct_blocks (list of numpy.ndarray): A list of 2D numpy arrays representing the DCT-transformed 
            image blocks. Each block should have a shape of (8, 8) with `float32` values.
        
            quant_matrix (numpy.ndarray): A predefined 8x8 quantization matrix that determines how each frequency 
            component is quantized.

        Returns:
            list of numpy.ndarray: A list of quantized image blocks, where each block is a 2D numpy array of shape (8, 8) 
            with integer values.
        """ 
        return [self.quantize(block, quant_matrix) for block in dct_blocks]

    def process_blocks_for_rle(self, blocks):
        """
        Processes a list of image blocks by applying zigzag scanning and Run-Length Encoding (RLE).

        This method performs the following steps for each block:
        1. Applies zigzag scanning to reorder the DCT coefficients.
        2. Encodes the reordered coefficients using Run-Length Encoding (RLE).

        Args:
            blocks (list of numpy.ndarray): A list of 2D numpy arrays, each of shape (8, 8), representing image blocks.

        Returns:
            list of lists: A list where each element is a list of RLE-encoded values. Each sublist corresponds to a block and contains tuples of (value, count) representing the RLE-encoded data.
        """
        # Apply zigzag scanning to each block
        zigzag_blocks = self.process_blocks_zigzag(blocks)
    
        # Apply Run-Length Encoding (RLE) to each zigzag-scanned block
        rle_blocks = self.process_blocks_rle(zigzag_blocks)
    
        return rle_blocks

    def process_blocks_zigzag(self, blocks):
        """
        Applies zigzag scanning to each block of image data.

        Zigzag scanning is used to reorder the DCT coefficients in a zigzag pattern, which helps in 
        grouping similar values together for more efficient compression.

        Args:
            blocks (list of numpy.ndarray): A list of 2D numpy arrays, each of shape (8, 8), representing image blocks 
            with DCT coefficients.

        Returns:
            list of numpy.ndarray: A list where each element is a 1D numpy array containing the reordered coefficients 
            of the corresponding block, arranged in a zigzag pattern.
        """
        # Apply zigzag scanning to each block
        zigzag_blocks = [self.zigzag_scan(block) for block in blocks]
        
        return zigzag_blocks

    def process_blocks_rle(self, zigzag_blocks):
        """
        Applies Run-Length Encoding (RLE) to each block of coefficients reordered by zigzag scanning.

        Run-Length Encoding is used to compress sequences of repeated values by storing the value and its count.

        Args:
            zigzag_blocks (list of list): A list of blocks where each block is a list of DCT coefficients reordered 
                                       in a zigzag pattern. Each block is a 1D list representing the coefficients 
                                       of an 8x8 image block.

        Returns:
            list of list: A list of blocks where each block is a list of tuples, each tuple contains a value and its 
                         count, representing the RLE-encoded coefficients of the corresponding zigzag-scanned block.
        """
        # Apply RLE encoding to each zigzag-scanned block
        rle_blocks = [self.run_length_encode(block) for block in zigzag_blocks]
        
        return rle_blocks


    def flatten_rle_blocks(self, rle_blocks):
        """
        Flattens a list of RLE-encoded blocks into a single list.

        This method takes a list of RLE-encoded blocks (where each block is itself a list of RLE-encoded data)
        and flattens it into a single list containing all the RLE-encoded data in sequence.
        
        Args:
            rle_blocks (list of list of tuples): A list of RLE-encoded blocks. Each block is a list of tuples, where each
                                             tuple contains a value and its count, representing RLE-encoded data.

        Returns:
            list of tuples: A single flattened list of RLE-encoded data, where each element is a tuple (value, count),
                        representing the concatenated RLE-encoded data from all input blocks.
        """
        return [item for sublist in rle_blocks for item in sublist]

     
    def reformat(self, image):
        """
        Reformats an image to ensure its dimensions are multiples of 8 by padding.

        This method adjusts the dimensions of the input image to be multiples of 8 by adding padding if necessary. 
        Padding is added using a reflective border mode to minimize edge artifacts.

        Args:
            image (numpy.ndarray): A 2D or 3D NumPy array representing the image. For a 2D array, it represents grayscale image data.
                               For a 3D array, it represents color image data with dimensions (height, width, channels).

        Returns:
            numpy.ndarray: A NumPy array representing the padded image with dimensions rounded up to the nearest multiple of 8.
        """
        height, width = image.shape
        row_add = height % 8
        col_add = width % 8

        # Pad right side and bottom side separately to avoid applying padding in one direction more than once
        if col_add != 0:
            image = np.pad(image, ((0, 0), (0, 8 - col_add)), mode='reflect')
        if row_add != 0:
            image = np.pad(image, ((0, 8 - row_add), (0, 0)), mode='reflect')
        return image

    def split_into_blocks(self, image, block_size=8):
        """
        Splits an image into non-overlapping blocks of a specified size.

        This method divides the input image into smaller blocks of size `block_size x block_size`. 
        It ensures that the dimensions of the image are divisible by the block size. 
        If the image dimensions are not compatible, the method will raise an assertion error.

        Args:
        image (numpy.ndarray): A 2D or 3D NumPy array representing the image. For a 2D array, it represents grayscale image data.
                               For a 3D array, it represents color image data with dimensions (height, width, channels).
        block_size (int): The size of the blocks to split the image into. Default is 8. It must be a divisor of both image dimensions.

        Returns:
            numpy.ndarray: A NumPy array of shape (num_blocks, block_size, block_size, channels) if the input image is 3D, 
                       or (num_blocks, block_size, block_size) if the input image is 2D, containing the image blocks.
        """
        height, width = image.shape
        
        assert height % block_size == 0, "Height must be divisible by block_size"
        assert width % block_size == 0, "Width must be divisible by block_size"
         # Extract blocks
        blocks = [image[y:y + block_size, x:x + block_size] for y in range(0, height, block_size) for x in range(0, width, block_size)]
        return np.array(blocks)

    def quantize(self, dct_block, quant_matrix):
        """
        Quantizes a Discrete Cosine Transform (DCT) block using a quantization matrix.

        This method performs quantization on a given DCT block by dividing each coefficient 
        by the corresponding value in the quantization matrix, and then rounding the result 
        to the nearest integer.

        Args:
            dct_block (numpy.ndarray): A 2D NumPy array representing a block of DCT coefficients. The block should be of shape (8, 8).
            quant_matrix (numpy.ndarray): A 2D NumPy array representing the quantization matrix. It should have the same shape as `dct_block`.

        Returns:
            numpy.ndarray: A 2D NumPy array of quantized DCT coefficients, with the same shape as the input `dct_block`. 
                       The quantized values are rounded to the nearest integer and are of integer type (int32).
        """
        return np.round(dct_block / quant_matrix).astype(np.int32)

    def subsample(self, channel):
        """
        Performs chroma subsampling on the input channel.

        This method reduces the resolution of the input channel by a factor of 2 in both dimensions.
        It achieves this by selecting every second pixel in both the horizontal and vertical directions.

        Args:
            channel (numpy.ndarray): A 2D NumPy array representing the image channel to be subsampled. 
                                 Typically used for color channels (e.g., Cb, Cr) in YCbCr color space.

        Returns:
            numpy.ndarray: A 2D NumPy array representing the subsampled image channel. The resolution of 
                       the channel is reduced by a factor of 2 in both dimensions.
        """
                             
        return channel[::2, ::2]
    
    def convert_numpy_to_int(self, data):
        """
        Converts elements of a nested list or array from float to integer.

        This method assumes that 'data' is a nested list (list of lists) where each sublist
        contains numerical values that need to be converted to integers.

        Args:
            data (list of lists or numpy.ndarray): A nested list or array-like structure containing numerical values.

        Returns:
            list of lists or numpy.ndarray: A nested list or array with all numerical values converted to integers.
        """
        return [[int(value) for value in block] for block in data]


    def run_length_encode(self, vector):
        """
        Encodes a list of values using Run-Length Encoding (RLE).

        This method compresses a list of values by representing consecutive 
        occurrences of the same value with a pair consisting of the value and 
        the count of its consecutive occurrences.

        Args:
            vector (list): A list of values to be encoded, where consecutive occurrences 
                       of the same value are represented as a single pair.

        Returns:
            list: A list of pairs, where each pair contains:
                - The value from the input list.
                - The count of consecutive occurrences of that value.
        """
        
        encoded = []
        prev_value = vector[0]
        count = 1
        for value in vector[1:]:
            if value == prev_value:
                count += 1
            else:
                encoded.append([prev_value, count])
                prev_value = value
                count = 1
        # Append the last run
        encoded.append([prev_value, count])
        return encoded

    def zigzag_scan(self, block):
        """
        Performs zigzag scanning on a 2D block.

        The zigzag scan traverses the 2D block in a zigzag order, which is commonly used in image compression, 
        such as JPEG. The traversal starts from the top-left corner and moves diagonally, alternating directions.

        Args:
            block (numpy.ndarray): A 2D numpy array representing the block to be scanned.

        Returns:
            list: A 1D list of values in zigzag order.
        """
        rows, cols = block.shape
        result = []
        for i in range(rows + cols - 1):
            if i % 2 == 0:
            # Moving upwards diagonally
                row, col = min(i, rows - 1), max(0, i - rows + 1)
                while row >= 0 and col < cols:
                    result.append(block[row, col])
                    row -= 1
                    col += 1
            else:
                # Moving downwards diagonally
                row, col = max(0, i - cols + 1), min(i, cols - 1)
                while col >= 0 and row < rows:
                    result.append(block[row, col])
                    row += 1
                    col -= 1
        return result
    
     
    def conv_rle_to_list(self, rle):
        """
        Converts RLE (Run-Length Encoding) data into a format suitable for Huffman encoding.

        This method processes a list of RLE pairs and transforms each pair into a format
        compatible with Huffman encoding. The first element of each RLE pair is adjusted
        by adding 128 to it, while the second element remains unchanged.

        Args:
            rle (list of tuples or lists): A list of RLE pairs, where each pair contains exactly two elements.
                                       The first element is a value, and the second element is its count.

        Returns:
            list: A list of integers formatted for Huffman encoding, where each value from the RLE pairs
              has 128 added to it.
        """
           
        huffman_format = []
        
        for pair in rle:
            if len(pair) != 2:
                raise ValueError("Each item in the RLE list must contain exactly two elements.")
            # Adjust the first element by adding 128
            huffman_format.append(pair[0] + 128)
            # Add the second element (count) as is
            huffman_format.append(pair[1])
        return huffman_format
    
   

    def decompress(self, compressed_huffman, huffman_table, length, height, luminance_length, luminance_height, blue_length, blue_height, red_length, red_height, len_rle_lum, len_rle_blue, len_rle_red):
        """
     Decompresses JPEG image data using Huffman coding and JPEG decompression steps.

        This method performs the following steps:
        1. Convert byte data to integer dimensions.
        2. Decompress Huffman-encoded data.
        3. Convert decompressed data from Huffman format to RLE.
        4. Decode RLE data back into blocks.
        5. Reverse zigzag scanning and apply IDCT to reconstruct image blocks.
        6. Combine blocks, adjust sizes, and reconstruct the RGB image.

        Args:
            compressed_huffman (bytes): Huffman-encoded compressed image data.
            huffman_table (dict): Table used for Huffman decoding.
            length (bytes): Image width as a 3-byte integer.
            height (bytes): Image height as a 3-byte integer.
            luminance_length (bytes): Luminance channel width as a 3-byte integer.
            luminance_height (bytes): Luminance channel height as a 3-byte integer.
            blue_length (bytes): Blue channel width as a 3-byte integer.
            blue_height (bytes): Blue channel height as a 3-byte integer.
            red_length (bytes): Red channel width as a 3-byte integer.
            red_height (bytes): Red channel height as a 3-byte integer.
            len_rle_lum (bytes): Length of RLE data for luminance channel as a 3-byte integer.
            len_rle_blue (bytes): Length of RLE data for blue channel as a 3-byte integer.
            len_rle_red (bytes): Length of RLE data for red channel as a 3-byte integer.

        Returns:
            numpy.ndarray: Reconstructed RGB image as a NumPy array.
        """
        # Convert byte data to integers
        width = int.from_bytes(length, byteorder='big')
        height = int.from_bytes(height, byteorder='big')
                
        luminance_length = int.from_bytes(luminance_length, byteorder='big')
        luminance_height = int.from_bytes(luminance_height, byteorder='big')
        
        blue_length = int.from_bytes(blue_length, byteorder='big')  
        blue_height = int.from_bytes(blue_height, byteorder='big')  

        red_length = int.from_bytes(red_length, byteorder='big')  
        red_height = int.from_bytes(red_height, byteorder='big')  
                
        rle_blue = int.from_bytes(len_rle_blue, byteorder='big')
        rle_red = int.from_bytes(len_rle_red, byteorder='big')
        rle_lum = int.from_bytes(len_rle_lum, byteorder='big')
        
        # Decompress Huffman data
        huffman = Huffman()
        decompressed_rle_list = huffman.decompress(compressed_huffman, huffman_table)
                
        # Convert decompressed Huffman data to RLE format
        compressed_rle = self.conv_list_to_rle(decompressed_rle_list)
        
        # Defining the RLE data length for each channel
        rle_lum_length = width * height
        rle_blue_length = blue_length * blue_height
        rle_red_length = red_length * red_height
        
        # Extract RLE data for each channel
        lum_rle = compressed_rle[:rle_lum]
        blue_rle = compressed_rle[rle_lum: rle_lum + rle_blue]
        red_rle = compressed_rle[rle_lum +rle_blue: rle_lum+rle_blue+rle_red]
              
        # Decode RLE data to blocks
        lum_data_blocks = self.run_length_decode(lum_rle) 
        blue_data_blocks = self.run_length_decode(blue_rle)
        red_data_blocks = self.run_length_decode(red_rle)
        
        # Reverse zigzag scan 
        lum_data_zigzag = self.process_blocks_from_rle(lum_data_blocks,8) 
        blue_data_zigzag = self.process_blocks_from_rle(blue_data_blocks,8) 
        red_data_zigzag = self.process_blocks_from_rle(red_data_blocks,8) 
                
        # Perform dequantization
        quant_lum_blocks_inverse = self.dequantize_blocks(lum_data_zigzag, self.quant_lum)
        quant_blue_blocks_inverse = self.dequantize_blocks(blue_data_zigzag, self.quant_colour)
        quant_red_blocks_inverse = self.dequantize_blocks(red_data_zigzag, self.quant_colour)
                
        # Apply IDCT to blocks
        idct_lum_blocks_inverse = self.idct_blocks(quant_lum_blocks_inverse)
        idct_blue_blocks_inverse = self.idct_blocks(quant_blue_blocks_inverse)
        idct_red_blocks_inverse = self.idct_blocks(quant_red_blocks_inverse)
                
        # Combine blocks and restore image size
        reconstructed_luminance = self.combine_blocks(idct_lum_blocks_inverse, luminance_length, luminance_height)
        reconstructed_blue = self.combine_blocks(idct_blue_blocks_inverse,blue_length, blue_height)
        reconstructed_red = self.combine_blocks(idct_red_blocks_inverse, red_length, red_height)
                      
        target_size = (luminance_length, luminance_height)
        blue_restore = self.restore_size(reconstructed_blue, target_size)
        red_restore = self.restore_size(reconstructed_red, target_size)

        # Convert YCbCr back to RGB        
        image_rgb_restore = self.ycbcr_to_rgb(reconstructed_luminance, blue_restore, red_restore)
        
        # Crop image to original dimensions
        image_rgb_restore = image_rgb_restore[:height, :width]
        
        return image_rgb_restore

  
    def ycbcr_to_rgb(self, y, cb, cr):
        """
        Converts YCbCr channels to RGB format.

        This method takes the Y (luminance), Cb (blue-difference chroma), and Cr (red-difference chroma)
        channels and combines them into an RGB image. The YCbCr to RGB conversion is based on the standard
        formula used in JPEG compression.

        :param y: 2D numpy array of luminance (Y) channel.
        :param cb: 2D numpy array of blue-difference chroma (Cb) channel.
        :param cr: 2D numpy array of red-difference chroma (Cr) channel.
        :return: 3D numpy array of RGB image with shape (height, width, 3).
        """
        # Initialize an empty RGB image with the same height and width as Y channel
        img_rgb = np.zeros((y.shape[0], y.shape[1], 3), dtype=np.uint8)

        # Convert YCbCr to RGB
        for row in range(y.shape[0]):
            for col in range(y.shape[1]):
                Y = y[row, col]
                Cb = cb[row, col]
                Cr = cr[row, col]

                # Convert YCbCr to RGB
                r = Y + 1.402 * (Cr - 128)
                g = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
                b = Y + 1.772 * (Cb - 128)
                # Clip values to the range [0, 255] and assign to the RGB image
                img_rgb[row, col, 0] = np.clip(r, 0, 255) # Red channel
                img_rgb[row, col, 1] = np.clip(g, 0, 255) # Green channel
                img_rgb[row, col, 2] = np.clip(b, 0, 255) # Blue channel

        return img_rgb

    
    def inverse_zigzag_scan(self, flat_sequence, block_size):
        """
        Reconstructs a matrix from a flat sequence using inverse zigzag scanning.

        Args:
            flat_sequence (list or np.ndarray): The flattened sequence of matrix elements 
                                            obtained from zigzag scanning.
            block_size (int): The size of the block (matrix is block_size x block_size).

        Returns:
            np.ndarray: The reconstructed matrix.
        """
        # Initialize an empty matrix of the specified block size
        matrix = np.zeros((block_size, block_size), dtype=np.int32)
        
        index = 0
        # Iterate over the diagonal indices of the matrix
        for sum_index in range(2 * (block_size - 1) + 1):
            if sum_index % 2 == 0:
                row_start = min(sum_index, block_size - 1)
                row_end = max(0, sum_index - (block_size - 1))
                for row in range(row_start, row_end - 1, -1):
                    col = sum_index - row
                    matrix[row][col] = flat_sequence[index]
                    index += 1
            else:
                # If the diagonal index is odd, traverse from top-right to bottom-left
                col_start = min(sum_index, block_size - 1)
                col_end = max(0, sum_index - (block_size - 1))
                for col in range(col_start, col_end - 1, -1):
                    row = sum_index - col
                    matrix[row][col] = flat_sequence[index]
                    index += 1

        return matrix

    def process_blocks_from_rle(self, decoded_data, block_size=8):
        """
        Converts RLE-decoded data back into blocks by performing inverse zigzag scanning.

        Args:
            decoded_data (list): RLE-decoded data as a flat list of values.
            block_size (int): The size of each block (typically 8x8 for JPEG).

        Returns:
            list: A list of 2D numpy arrays (blocks) where each block is of size (block_size, block_size).
        """
        # Calculate the number of blocks
        num_blocks = len(decoded_data) // (block_size * block_size)
        blocks = []
        for i in range(num_blocks):
            start = i * block_size * block_size
            end = start + block_size * block_size
            block_data = decoded_data[start:end]
            
            # Convert the flat sequence back to a block using inverse zigzag scan
            matrix = self.inverse_zigzag_scan(block_data, block_size)
            blocks.append(matrix)
        return blocks
    
    def dequantize_blocks(self, quant_blocks, quant_matrix):
        """
        Applies dequantization to a list of quantized DCT blocks using a quantization matrix.

        Args:
            quant_blocks (list): A list of 2D numpy arrays where each array is a quantized DCT block.
            quant_matrix (numpy array): A 2D numpy array representing the quantization matrix.

        Returns:
            list: A list of 2D numpy arrays where each array is a dequantized DCT block.
        """
        return [self.dequantize(block, quant_matrix) for block in quant_blocks]
    
    def dequantize(self, quant_block, quant_matrix):
        """
        Applies dequantization to a quantized DCT block using the quantization matrix.

        Args:
            quant_block (numpy.ndarray): A 2D numpy array representing the quantized DCT block.
            quant_matrix (numpy.ndarray): The quantization matrix used during compression.

        Returns:
            numpy.ndarray: A 2D numpy array representing the dequantized DCT block.
        """
        return quant_block * quant_matrix
    
    def idct_blocks(self, quant_blocks_reverse):
         """
        Applies the Inverse Discrete Cosine Transform (IDCT) to a list of quantized DCT blocks.

        Args:
            quant_blocks_reverse (list of numpy.ndarray): A list of 2D numpy arrays representing the quantized DCT blocks.

        Returns:
            list of numpy.ndarray: A list of 2D numpy arrays representing the blocks after IDCT is applied.
        """
         return [self.idct(block) for block in quant_blocks_reverse]
    
    def idct(self, quant_block_reverse):
        """
        Applies the Inverse Discrete Cosine Transform (IDCT) to a single quantized DCT block.

        Args:
            quant_block_reverse (numpy.ndarray): A 2D numpy array representing a quantized DCT block.

        Returns:
            numpy.ndarray: A 2D numpy array representing the block after IDCT is applied.
        """
        return cv2.idct(quant_block_reverse.astype(np.float32))

    def combine_blocks(self, blocks, width, height):
        """
        Combines a list of image blocks into a single image.

        Args:
            blocks (list of numpy.ndarray): List of 2D numpy arrays representing image blocks.
            width (int): The width of the final combined image.
            height (int): The height of the final combined image.

        Returns:
            numpy.ndarray: The combined image as a 2D numpy array.
        """
        # Define the block size
        block_size = 8
        
        # Create an empty matrix for the combined image
        combined_image = np.zeros((height, width), dtype=np.float32)
        
        # Determine the number of blocks along x and y directions
        num_blocks_x = width // block_size
        num_blocks_y = height // block_size
        # Fill the image with blocks
        for i in range(num_blocks_y):
            for j in range(num_blocks_x):
                y = i * block_size
                x = j * block_size
                combined_image[y:y+block_size, x:x+block_size] = blocks[i * num_blocks_x + j]
                
        return combined_image


    def restore_size(self, channel, target_size):
        """
        Resizes the given channel to the target size.

        Args:
            channel (numpy.ndarray): The image channel (2D array) to resize.
            target_size (tuple): The desired size as (width, height).

        Returns:
            numpy.ndarray: The resized channel.
        """
        return cv2.resize(channel, target_size, interpolation=cv2.INTER_LINEAR)
  
    def run_length_decode(self, rle_encoded_data):
        """
        Decodes RLE (Run-Length Encoded) data back into a list of values.

        Args:
            rle_encoded_data (list of tuples): The RLE encoded data, where each tuple contains a value and its run length.

        Returns:
            list: The decoded data as a list of values.
        """
        decoded_data = []
        for value, length in rle_encoded_data:
            decoded_data.extend([value] * length)
        return decoded_data

    def conv_list_to_rle(self, data):
        """
        Converts a list of integers into Run-Length Encoding (RLE) format.

        Args:
            data (list of int): The list of integers to convert. 
                                Each pair of values represents a run length encoding.
                                The first value in each pair is adjusted by subtracting 128.

        Returns:
            list of lists: The RLE encoded data, where each sublist contains a value and its run length.
        """
        rle = []
        for i in range(0, len(data), 2):
            rle.append([data[i] - 128, data[i + 1]])
        return rle
