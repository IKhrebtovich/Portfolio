import unittest
from unittest.mock import patch, mock_open, MagicMock
import tkinter as tk
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/algorithms')))
from src.compression_app import CompressionApp


class TestCompressionOperation(unittest.TestCase):
    def setUp(self):
        """
        Prepare test data. Create an instance of CompressionApp and set
        initial values for parameters.
        """
        self.app = CompressionApp()
        self.app.original_extension = 'txt'
        self.app.decompressed_data = b'decompressed data'
        self.app.current_compressed_data = [1, 2, 3, 4]  
        self.app.current_codebook = {b'\x01': 1, b'\x02': 2}  
        self.app.current_compressed_data = b'compressed_data'
        self.app.current_codebook = {1: '0001', 2: '0010'}
       
        self.app.length = b'length_data'
        self.app.height = b'height_data'
        self.app.luminance_length = b'luminance_length_data'
        self.app.luminance_height = b'luminance_height_data'
        self.app.blue_length = b'blue_length_data'
        self.app.blue_height = b'blue_height_data'
        self.app.red_length = b'red_length_data'
        self.app.red_height = b'red_height_data'
        self.app.len_rle_lum = b'len_rle_lum_data'
        self.app.len_rle_blue = b'len_rle_blue_data'
        self.app.len_rle_red = b'len_rle_red_data' 
    
    @patch('src.compression_app.filedialog.asksaveasfilename', return_value='test_file.lzw')
    @patch('src.compression_app.open', new_callable=mock_open)
    @patch('src.compression_app.messagebox.showinfo')
    def test_save_compressed_lzw(self, mock_showinfo, mock_open, mock_asksaveasfilename):
        """
        Test the function for saving a compressed LZW file. Ensure that
        files are opened with the correct paths and that a success message
        is displayed.
        """
        # Setup mock data
        self.app.current_compressed_data = [1, 2, 3, 4]  
        self.app.current_codebook = {b'key': 1234}  
        self.app.original_extension = '.txt' 
        
        self.app.save_compressed_file_lzw()
        
        # Verify files are opened with the correct paths
        mock_open.assert_any_call('test_file.lzw', 'wb')
        mock_open.assert_any_call('test_file.lzw.codebook', 'wb')
        mock_open.assert_any_call('test_file.lzw.ext', 'w')
        
        # Verify success message is shown
        mock_showinfo.assert_called_once_with('Success', 'Compressed file and codebook saved successfully.')

    
    @patch('src.compression_app.filedialog.asksaveasfilename', return_value='test_file.lzw')
    @patch('src.compression_app.open', new_callable=mock_open)
    @patch('src.compression_app.messagebox.showerror')
    def test_save_compressed_lzw_save_error(self, mock_showerror, mock_open, mock_asksaveasfilename):
        """
        Test the behavior when an exception occurs while saving files. An error message should be displayed.
        """
        self.app.current_compressed_data = [1, 2, 3]  
        self.app.current_codebook = {'key1': 1, 'key2': 2}
        self.app.original_extension = '.txt'

        # Simulate an error when opening the file
        mock_open.side_effect = IOError('File save error')

        self.app.save_compressed_file_lzw()

        # Verify that error message is shown
        mock_showerror.assert_called_once_with('Error', 'Failed to save files: File save error')

    @patch('src.compression_app.filedialog.asksaveasfilename', return_value=None)
    @patch('src.compression_app.messagebox.showerror')
    def test_save_compressed_lzw_no_data(self, mock_showerror, mock_asksaveasfilename):
        """
        Test the behavior when there is no compressed data to save. An error message should be displayed.
        """
        self.app.current_compressed_data = None  
        self.app.current_codebook = None
        self.app.original_extension = '.txt'

        self.app.save_compressed_file_lzw()

        mock_showerror.assert_called_once_with('Error', 'No compressed data to save')

    @patch('src.compression_app.filedialog.asksaveasfilename', return_value='')
    @patch('src.compression_app.messagebox.showwarning')
    def test_save_compressed_lzw_no_path_selected(self, mock_showwarning, mock_asksaveasfilename):
        """
        Test the behavior of saving a compressed LZW file when no file path
        is selected. A warning message should be displayed.
        """
        self.app.save_compressed_file_lzw()

        mock_showwarning.assert_called_once_with('Warning', 'No file path selected')

    @patch('src.compression_app.filedialog.asksaveasfilename', return_value='test_file.lzw')
    @patch('src.compression_app.open', new_callable=mock_open)
    @patch('src.compression_app.messagebox.showinfo')
    @patch('src.compression_app.messagebox.showerror')
    @patch('src.compression_app.messagebox.showwarning')
    def test_save_decompressed_file_lzw(self, mock_showwarning, mock_showerror, mock_showinfo, mock_open, mock_asksaveasfilename):
        """
        Test the function for saving a decompressed LZW file. Ensure that
        the file is saved with the correct data and that a success message
        is displayed.
        """
        self.app.save_decompressed_file_lzw()

        mock_open.assert_called_once_with('test_file.lzw', 'wb')
        mock_open().write.assert_called_once_with(b'decompressed data')
        mock_showinfo.assert_called_once_with("Success", "Decompressed file successfully saved")
        mock_showerror.assert_not_called()
        mock_showwarning.assert_not_called()

    @patch('src.compression_app.filedialog.asksaveasfilename', return_value='test_file.lzw')
    @patch('src.compression_app.open', new_callable=mock_open)
    @patch('src.compression_app.messagebox.showerror')
    def test_save_decompressed_file_lzw_exception(self, mock_showerror, mock_open, mock_asksaveasfilename):
        """
        Test the behavior of saving a decompressed LZW file when an exception occurs.
        An error message should be displayed.
        """
        self.app.decompressed_data = b'decompressed data'
        self.app.original_extension = '.lzw'

        # Simulate an exception when opening the file
        mock_open.side_effect = IOError("Failed to save file")

        # Call the method
        self.app.save_decompressed_file_lzw()

        # Verify error message is shown
        mock_showerror.assert_called_once_with("Error", "Failed to save file: Failed to save file")
      

    @patch('src.compression_app.messagebox.showinfo')
    @patch('src.compression_app.LZW.compress')
    @patch('src.compression_app.FileHandler.read_file')
    
    def test_compress_file_lzw(self, mock_read_file, mock_compress, mock_showinfo):
        """
        Test the function for compressing a file using LZW. Ensure that
        the data is compressed correctly and that a success message
        is displayed.
        """
        mock_read_file.return_value = b'Test data'
        mock_compress.return_value =([1, 2, 3], {'key': 'value'})

        result = self.app.compress_file_lzw('test.txt')

        self.assertTrue(result)
        self.assertEqual(self.app.current_codebook, {'key': 'value'})
        mock_showinfo.assert_called_once_with('Compression results', 'File successfully compressed')

    @patch('src.compression_app.messagebox.showinfo')
    @patch('src.compression_app.LZW.compress')
    @patch('src.compression_app.FileHandler.read_file')
    def test_compress_file_lzw_exception(self, mock_read_file, mock_compress, mock_showinfo):
        """
        Test the function for compressing a file using LZW when an exception occurs.
        Ensure that an error message is displayed.
        """
        # Setup mocks
        mock_read_file.return_value = b'Test data'
        mock_compress.side_effect = Exception("Compression error")

        # Call the method
        result = self.app.compress_file_lzw('test.txt')

        # Check that the method returned False
        self.assertFalse(result)

        # Check that the error message was displayed
        mock_showinfo.assert_called_once_with('Compression Error', 'Failed to compress file: Compression error')

    @patch('src.compression_app.messagebox.showerror')
    @patch('src.compression_app.messagebox.showinfo')
    @patch('src.compression_app.LZW.decompress')
    @patch('src.compression_app.FileHandler.load_codebook_lzw')
    @patch('src.compression_app.os.path.exists')
    @patch('src.compression_app.open', new_callable=mock_open, read_data=b'compressed_data')
    def test_decompress_file_lzw(self, mock_open, mock_exists, mock_load_codebook, mock_decompress, mock_showinfo, mock_showerror):
        """
        Test the function for decompressing an LZW file. Ensure that
        the file is decompressed correctly and that a success message
        is displayed.
        """
        # Setting up mocks
        mock_exists.side_effect = [True, True, True]
        
        mock_load_codebook.return_value = {'key': 'value'}
        mock_decompress.return_value = b'decompressed data'
        
        # Calling the method under test
        result = self.app.decompress_file_lzw('test.lzw')

        self.assertTrue(result)
        self.assertEqual(self.app.decompressed_data, b'decompressed data')
        mock_showinfo.assert_called_once_with('Decompression results', 'File successfully decompressed')
        mock_showerror.assert_not_called()

    @patch('src.compression_app.messagebox.showerror')
    def test_decompress_file_lzw_invalid_extension(self, mock_showerror):
        """
        Test the behavior when the provided file path does not have the .lzw extension.
        An error message should be displayed.
        """
        file_path = 'test.txt'  

        result = self.app.decompress_file_lzw(file_path)

        self.assertFalse(result)
        mock_showerror.assert_called_once_with("Decompression Error", "Selected file does not have the .lzw extension.")

    @patch('src.compression_app.messagebox.showerror')
    @patch('src.compression_app.os.path.exists')
    def test_decompress_file_lzw_files_not_found(self, mock_exists, mock_showerror):
        """
        Test the behavior when the compressed file or codebook is missing.
        An error message should be displayed.
        """
        file_path = 'test.lzw'
        
        # Simulating the absence of files
        mock_exists.side_effect = [False, False, True]  # file and code book missing

        result = self.app.decompress_file_lzw(file_path)

        self.assertFalse(result)
        mock_showerror.assert_called_once_with("Decompression Error", "Compressed file or codebook not found.")
    

    @patch('src.compression_app.filedialog.asksaveasfilename', return_value = 'test.txt.huff')
    @patch('src.compression_app.open', new_callable=mock_open)
    @patch('src.compression_app.messagebox.showinfo')
    @patch('src.compression_app.messagebox.showerror')
    @patch('src.compression_app.messagebox.showwarning')
    def test_save_compressed_file_huffman(self, mock_showwarning, mock_showerror, mock_showinfo, mock_open, mock_asksaveasfilename):
        """
        Test the function for saving a compressed Huffman file. Ensure that
        the file is saved correctly and that a success message is displayed.
        """
        self.app.current_compressed_data = b'compressed_data'
        self.app.current_codebook = {1: '101010', 2: '111000'}
        self.app.original_extension = 'txt'

        self.app.save_compressed_file_huffman()

        mock_open.assert_any_call('test.txt.huff', 'wb')
        mock_showinfo.assert_called_once_with("Compression results", 'Compressed file and codebook saved successfully.')
        mock_showerror.assert_not_called()
        mock_showwarning.assert_not_called()
    
    @patch('src.compression_app.filedialog.asksaveasfilename', return_value='')
    @patch('src.compression_app.open', new_callable=mock_open)
    @patch('src.compression_app.messagebox.showwarning')
    def test_save_compressed_file_huffman_no_path_selected(self, mock_showwarning, mock_open, mock_asksaveasfilename):
        """
        Test the behavior of saving a compressed Huffman file when no file path
        is selected. A warning message should be displayed.
        """
        self.app.current_compressed_data = b'some compressed data'
        self.app.current_codebook = {1: '0000'}

        result = self.app.save_compressed_file_huffman()

        mock_open.assert_not_called()  
        mock_showwarning.assert_called_once_with('Warning', "No file path selected")
     
    @patch('src.compression_app.filedialog.asksaveasfilename', return_value = 'test_file.huff')
    @patch('src.compression_app.open', new_callable=mock_open)
    @patch('src.compression_app.messagebox.showerror')
    def test_save_compressed_file_huffman_exception(self, mock_showerror, mock_open, mock_asksaveasfilename):
        """
        Test handling exceptions when saving a compressed Huffman file.
        Ensure that an error message is displayed when there is a disk write error.
        """
        mock_open.side_effect = Exception('Disk write error')

        self.app.current_compressed_data = b'compressed_data'
        self.app.current_codebook = {1: '101010', 2: '111000'}
        self.app.original_extension = 'txt'
        
        self.app.save_compressed_file_huffman()

        mock_open.assert_called_once_with('test_file.huff', 'wb')                              
        mock_showerror.assert_called_once_with('Error', 'Failed to save files: Disk write error')
    
    @patch('src.compression_app.filedialog.asksaveasfilename', return_value = 'test_file.huff.txt')
    @patch('src.compression_app.open', new_callable=mock_open)
    @patch('src.compression_app.messagebox.showinfo')
    @patch('src.compression_app.messagebox.showerror')
    @patch('src.compression_app.messagebox.showwarning')
    def test_save_decompressed_file_huffman(self, mock_showwarning, mock_showerror, mock_showinfo, mock_open, mock_asksaveasfilename):
        """
        Test the function for saving a decompressed Huffman file. Ensure that
        the file is saved correctly and that a success message is displayed.
        """       
        self.app.save_decompressed_file_huffman()

        mock_open.assert_called_once_with('test_file.huff.txt', 'wb')
        mock_open().write.assert_called_once_with(b'decompressed data')
        mock_showinfo.assert_called_once_with("Success", "Decompressed file successfully saved")
        mock_showerror.assert_not_called()
        mock_showwarning.assert_not_called()

    @patch('src.compression_app.filedialog.asksaveasfilename', return_value='')
    @patch('src.compression_app.messagebox.showwarning')
    def test_save_decompressed_file_huffman_no_path_selected(self, mock_showwarning, mock_asksaveasfilename):
        """
        Test the behavior of saving a decompressed Huffman file when no file path
        is selected. A warning message should be displayed.
        """
        self.app.decompressed_data = b'decompressed data'  
        self.app.save_decompressed_file_huffman()

        mock_showwarning.assert_called_once_with("Warning", "File path not selected")
       

    @patch('src.compression_app.filedialog.asksaveasfilename')
    @patch('src.compression_app.open', new_callable=mock_open)
    @patch('src.compression_app.messagebox.showerror')                                          
    def test_save_decompressed_file_huffman_exception(self, mock_showerror, mock_open, mock_asksaveasfilename):
        """
        Test handling exceptions when saving a decompressed Huffman file.
        Ensure that an error message is displayed when there is a disk write error.
        """
        mock_asksaveasfilename.return_value = 'test.txt'
        mock_open.side_effect = Exception('Disk write error')

        self.app.save_decompressed_file_huffman()

        mock_open.assert_called_once_with('test.txt', 'wb')
        mock_showerror.assert_called_once_with("Error", 'Failed to save file: Disk write error')

 
    @patch('src.compression_app.FileHandler.read_file')
    @patch('src.compression_app.Huffman.compress')
    @patch('src.compression_app.messagebox.showinfo')
    @patch('src.compression_app.messagebox.showwarning')
    def test_compress_file_huffman(self, mock_showwarning, mock_showinfo, mock_compress, mock_read_file):
        """
        Test the function for compressing a file using Huffman coding. Ensure that
        the data is compressed correctly and that a success message is displayed.
        """
        mock_read_file.return_value = b"Test data"
        mock_compress.return_value = (b'compressed_data', {'key': 'value'})

        result = self.app.compress_file_huffman('test.txt')

        self.assertTrue(result)
        self.assertEqual(self.app.current_codebook, {'key': 'value'})
        mock_showinfo.assert_called_once_with('Compression results', 'File successfully compressed')
        mock_showwarning.assert_not_called()

    @patch('src.compression_app.FileHandler.load_codebook_huffman')
    @patch('src.compression_app.Huffman.decompress')
    @patch('src.compression_app.messagebox.showinfo')
    @patch('src.compression_app.messagebox.showerror')
    @patch('src.compression_app.os.path.exists')
    @patch('src.compression_app.open', new_callable=mock_open, read_data=b'compressed_data')
    def test_decompress_huffman(self, mock_open, mock_exists, mock_showerror, mock_showinfo, mock_decompress, mock_load_codebook_huffman):
        """
        Test the function for decompressing a Huffman file. Ensure that
        the file is decompressed correctly and that a success message
        is displayed.
        """
        mock_exists.side_effect = [True, True, True]
        mock_load_codebook_huffman.return_value = {'key': 'value'}
        mock_decompress.return_value = b'Decompressed data'
        self.app.decompressed_data = b''
        result = self.app.decompress_file_huffman('test.huff')

        self.assertTrue(result)
        self.assertEqual(self.app.decompressed_data, b'Decompressed data')
        mock_showinfo.assert_called_once_with('Decompression results', 'File successfully decompressed')
        mock_showerror.assert_not_called()
    
    @patch('src.compression_app.messagebox.showerror')
    def test_decompress_huffman_wrong_extension(self, mock_showerror):
        """
        Test the behavior when the file does not have the correct .huff extension.
        """
        result = self.app.decompress_file_huffman('test.txt')

        self.assertFalse(result)
        mock_showerror.assert_called_once_with("Decompression error", "Selected file does not have the .huff extension.")

    @patch('src.compression_app.os.path.exists')
    @patch('src.compression_app.messagebox.showerror')
    def test_decompress_huffman_files_not_found(self, mock_showerror, mock_exists):
        """
        Test the behavior when the compressed file or codebook is not found.
        """
        mock_exists.side_effect = [False, True, True]

        result = self.app.decompress_file_huffman('test.huff')

        self.assertFalse(result)
        mock_showerror.assert_called_once_with("Decompression error", "Compressed file or codebook not found.")

    @patch('src.compression_app.FileHandler.load_codebook_huffman')
    @patch('src.compression_app.Huffman.decompress')
    @patch('src.compression_app.messagebox.showerror')
    @patch('src.compression_app.open', new_callable=mock_open, read_data=b'compressed_data')
    @patch('src.compression_app.os.path.exists')
    def test_decompress_huffman_exception(self, mock_exists, mock_open, mock_showerror, mock_decompress, mock_load_codebook_huffman):
        """
        Test handling exceptions when decompressing a Huffman file.
        Ensure that an error message is displayed when there is a decompression error.
        """
        mock_exists.side_effect = [True, True, True] 
        mock_load_codebook_huffman.return_value = {'key': 'value'}
        mock_decompress.side_effect = Exception('Decompression failed')

        result = self.app.decompress_file_huffman('test.huff')

        self.assertFalse(result)
        mock_showerror.assert_called_once_with("Decompression error", "Error during decompression: Decompression failed")

    @patch('tkinter.filedialog.askopenfilename')
    def test_browse_path(self, mock_askopenfilename):
        # Setting up a mock object
        mock_askopenfilename.return_value = '/path/to/selected/file.txt'

        # Setting initial values ​​in a combo box
        self.app.file_combobox['values'] = ('existing_file.txt',)  
  
        self.app.browse_path()

        expected_values = ['existing_file.txt', '/path/to/selected/file.txt']

        # Getting the combobox values ​​and converting them to a list for comparison
        actual_values = list(self.app.file_combobox['values'])

        self.assertEqual(actual_values, expected_values)
        self.assertEqual(self.app.file_combobox.get(), '/path/to/selected/file.txt')


    @patch('src.compression_app.filedialog.asksaveasfilename', return_value='test_file.huff')
    @patch('src.compression_app.open', new_callable=mock_open)
    @patch('src.compression_app.messagebox.showinfo')
    def test_save_compressed_file_jpeg_success(self, mock_showinfo, mock_open, mock_asksaveasfilename):
        """
        Test the function for saving a compressed JPEG file. Ensure that
        files are opened with the correct paths and that a success message
        is displayed.
        """
        self.app.save_compressed_file_jpeg()

        # Verify files are opened with the correct paths
        mock_open.assert_any_call('test_file.huff', 'wb')
        mock_open.assert_any_call('test_file.huff.codebook', 'wb')
        mock_open.assert_any_call('test_file.huff.ext', 'w')
        mock_open.assert_any_call('test_file.huff.additional', 'wb')
        
        # Verify success message is shown
        mock_showinfo.assert_called_once_with("Compression results", 'Compressed JPEG file and codebook saved successfully.')

    
    @patch('src.compression_app.filedialog.asksaveasfilename', return_value='test_file.huff')
    @patch('src.compression_app.open', new_callable=mock_open)
    @patch('src.compression_app.messagebox.showerror')
    def test_save_compressed_file_jpeg_failure(self, mock_showerror, mock_open, mock_asksaveasfilename):
        """
        Test the function when an exception occurs during file saving. Ensure that
        an error message is displayed.
        """
        mock_open.side_effect = Exception("Test Exception")

        self.app.save_compressed_file_jpeg()
        
        # Verify error message is shown
        mock_showerror.assert_called_once_with('Error', 'Failed to save files: Test Exception')

    @patch('src.compression_app.filedialog.asksaveasfilename', return_value='test_file.huff')
    @patch('src.compression_app.open', new_callable=mock_open)
    @patch('src.compression_app.messagebox.showerror')
    def test_save_compressed_file_jpeg_no_data(self, mock_showerror, mock_open, mock_asksaveasfilename):
        """
        Test the function when no compressed data is available. Ensure that
        an error message is displayed.
        """
        self.app.current_compressed_data = None
        self.app.current_codebook = None

        self.app.save_compressed_file_jpeg()
        
        # Verify error message is shown
        mock_showerror.assert_called_once_with('Error', 'No compressed data to save')

    @patch('src.compression_app.filedialog.asksaveasfilename', return_value='test_file.png')
    @patch('src.compression_app.open', new_callable=mock_open)
    @patch('src.compression_app.cv2.imwrite')
    @patch('src.compression_app.messagebox.showinfo')
    def test_save_decompressed_file_jpeg_success(self, mock_showinfo, mock_cv2_imwrite, mock_open, mock_asksaveasfilename):
        """
        Test the function for saving a decompressed JPEG file. Ensure that
        the file is saved correctly and a success message is displayed.
        """
        self.app.save_decompressed_file_jpeg()

        # Verify that the file is opened with the correct path
        mock_open.assert_called_once_with('test_file.png', 'wb')
        
        # Verify that cv2.imwrite is called with the correct arguments
        mock_cv2_imwrite.assert_called_once_with('test_file.png', self.app.decompressed_data)
        
        # Verify success message is shown
        mock_showinfo.assert_called_once_with("Success", "Image successfully saved as PNG")

    @patch('src.compression_app.filedialog.asksaveasfilename', return_value='test_file.png')
    @patch('src.compression_app.open', new_callable=mock_open)
    @patch('src.compression_app.cv2.imwrite')
    @patch('src.compression_app.messagebox.showerror')
    def test_save_decompressed_file_jpeg_failure(self, mock_showerror, mock_cv2_imwrite, mock_open, mock_asksaveasfilename):
        """
        Test the function when an exception occurs during file saving. Ensure that
        an error message is displayed.
        """
        mock_cv2_imwrite.side_effect = Exception("Test Exception")

        self.app.save_decompressed_file_jpeg()
        
        # Verify error message is shown
        mock_showerror.assert_called_once_with("Error", "Failed to save image: Test Exception")

    @patch('src.compression_app.FileHandler.read_image_file', return_value=b'some_image_data')
    @patch('src.compression_app.JPEG.compress', return_value=(b'compressed_data', {}, 100, 200, 10, 20, 30, 40, 50, 60, 70, 80, 90))
    @patch('src.compression_app.messagebox.showinfo')
    def test_compress_file_jpeg_success(self, mock_showinfo, mock_jpeg_compress, mock_read_image_file):
        """
        Test the function for successful JPEG compression. Ensure that
        compression is performed and a success message is displayed.
        """
        result = self.app.compress_file_jpeg('test_file.jpg')

        # Verify compression result
        self.assertTrue(result)
        
        # Verify that the file was read
        mock_read_image_file.assert_called_once_with('test_file.jpg')
        
        # Verify that JPEG.compress was called with correct data
        mock_jpeg_compress.assert_called_once_with(b'some_image_data')
        
        # Verify that success message is shown
        mock_showinfo.assert_called_once_with('Compression results', 'File successfully compressed')

    @patch('src.compression_app.FileHandler.read_image_file', return_value=b'some_image_data')
    @patch('src.compression_app.JPEG.compress', side_effect=Exception("Compression error"))
    @patch('src.compression_app.messagebox.showerror')
    def test_compress_file_jpeg_failure(self, mock_showerror, mock_jpeg_compress, mock_read_image_file):
        """
        Test the function when an exception occurs during compression. Ensure that
        an error message is displayed.
        """
        result = self.app.compress_file_jpeg('test_file.jpg')

        # Verify compression result
        self.assertFalse(result)
        
        # Verify that the error message is shown
        mock_showerror.assert_called_once_with('Compression Error', 'An error occurred during compression: Compression error')

    @patch('src.compression_app.filedialog.askopenfilename', return_value='test_file.txt')
    @patch('src.compression_app.messagebox.showerror')
    def test_decompress_file_jpeg_wrong_extension(self, mock_showerror, mock_askopenfilename):
        result = self.app.decompress_file_jpeg('test_file.txt')

        self.assertFalse(result)
        mock_showerror.assert_called_once_with("Decompression error", "Selected file does not have the .huff extension.")


    @patch('src.compression_app.open', new_callable=mock_open)
    @patch('src.compression_app.FileHandler.load_codebook_huffman', side_effect=FileNotFoundError("File not found"))
    @patch('src.compression_app.messagebox.showerror')
    def test_decompress_file_jpeg_missing_files(self, mock_showerror, mock_load_codebook, mock_open):
        """ Test the function when the required files are missing."""
        result = self.app.decompress_file_jpeg('missing_file.huff')

        self.assertFalse(result)
        
        # Check that the error message appears
        mock_showerror.assert_called_once_with("Decompression error", "Compressed file or codebook not found.")


    @patch('src.compression_app.filedialog.askopenfilename', return_value='test_file.huff')
    @patch('src.compression_app.open', new_callable=mock_open)
    @patch('src.compression_app.FileHandler.load_codebook_huffman', return_value={1: '0001'})
    @patch('src.compression_app.JPEG.decompress', side_effect=Exception("Decompression error"))
    @patch('src.compression_app.messagebox.showerror')
    def test_decompress_file_jpeg_failure(self, mock_showerror, mock_jpeg_decompress, mock_load_codebook, mock_open, mock_askopenfilename):
        """ Test the function when an exception occurs during decompression."""
        result = self.app.decompress_file_jpeg('test_file.huff')

        self.assertFalse(result)

        mock_showerror.assert_called_once_with("Decompression error", "Compressed file or codebook not found.")

    


if __name__=='__main__':
    unittest.main()

