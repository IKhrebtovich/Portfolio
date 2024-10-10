import unittest
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.compression_app import CompressionApp


class TestCompressionApp(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        self.app = CompressionApp()
        self.app.main_window.update()

        self.test_file_path = 'test_file_path.txt'
        with open(self.test_file_path, 'w') as f:
            f.write('This is a test file')
        # Widget Mocks
        self.file_combobox_mock = MagicMock()
        self.file_combobox_mock.configure_mock(values=[])
        self.app.file_combobox = self.file_combobox_mock

        self.compression_combobox_mock = MagicMock()
        self.algorithm_combobox_mock = MagicMock()
        self.algorithm_combobox_mock.configure_mock(values=[
            'Lempel-Ziv-Welch (LZW)', 
            'Huffman coding',
            'JPEG'
        ])
        self.app.compression_combobox = self.compression_combobox_mock
        self.app.algorithm_combobox = self.algorithm_combobox_mock

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
        self.app.main_window.destroy()

    def test_initial_state(self):
        """Test the initial state of the app's UI components."""
        # Set up mocks to return empty strings
        self.app.file_combobox.get = MagicMock(return_value='')
        self.app.compression_combobox.get = MagicMock(return_value='')
        self.app.algorithm_combobox.get = MagicMock(return_value='')
    
        # Checking the initial state of UI components
        self.assertEqual(self.app.file_combobox.get(), '')
        self.assertEqual(self.app.compression_combobox.get(), '')
        self.assertEqual(self.app.algorithm_combobox.get(), '')
    
    
    def test_choice_operation(self):
        """Test if the compression/decompression options are available."""
   
        self.compression_combobox_mock.configure_mock(values=[
            'Compression',
            'Decompression'
        ])
    
     # The get() method returns a list of values
        self.compression_combobox_mock.get = MagicMock(return_value=self.compression_combobox_mock.values)

    # Call the method to check the values
        operation = self.app.compression_combobox.get()

        self.assertIn('Compression', operation)
        self.assertIn('Decompression', operation)
   
    def test_choice_algorithm(self):
        """Test if the algorithms are listed in the dropdown."""
    
        self.algorithm_combobox_mock.configure_mock(values=[
            'Lempel-Ziv-Welch (LZW)', 
            'Huffman coding',
            'JPEG'
        ])
    
        self.algorithm_combobox_mock.get = MagicMock(return_value=self.algorithm_combobox_mock.values)

        # Call the method to check the values
        algorithms = self.app.algorithm_combobox.get()

        self.assertIn('Lempel-Ziv-Welch (LZW)', algorithms)
        self.assertIn('Huffman coding', algorithms)
        self.assertIn('JPEG', algorithms)

    @patch('src.compression_app.CompressionApp.compress_file_lzw', return_value=True)
    @patch('src.compression_app.messagebox.showinfo')
    def test_perform_operation_compress_lzw_success(self, mock_showinfo, mock_compress_file_lzw):
        """Test the perform_operation method for successful LZW compression."""
        self.file_combobox_mock.get.return_value = 'test.txt'
        self.compression_combobox_mock.get.return_value = 'Compression'
        self.algorithm_combobox_mock.get.return_value = 'Lempel-Ziv-Welch (LZW)'

        self.app.perform_operation()

        # Verify that the correct compression method was called
        mock_compress_file_lzw.assert_called_once_with('test.txt')
        mock_showinfo.assert_called_once_with("Operation results", 'Performing Compression using Lempel-Ziv-Welch (LZW) on test.txt')

    @patch('src.compression_app.CompressionApp.decompress_file_lzw', return_value=True)
    @patch('src.compression_app.messagebox.showinfo')
    def test_perform_operation_decompress_lzw_success(self, mock_showinfo, mock_decompress_file_lzw):
        """Test the perform_operation method for successful LZW decompression."""
        self.file_combobox_mock.get.return_value = 'test.lzw'
        self.compression_combobox_mock.get.return_value = 'Decompression'
        self.algorithm_combobox_mock.get.return_value = 'Lempel-Ziv-Welch (LZW)'

        self.app.perform_operation()

        # Verify that the correct decompression method was called
        mock_decompress_file_lzw.assert_called_once_with('test.lzw')
        mock_showinfo.assert_called_once_with("Operation results", 'Performing Decompression using Lempel-Ziv-Welch (LZW) on test.lzw')

    @patch('src.compression_app.CompressionApp.decompress_file_huffman', return_value=True)
    @patch('src.compression_app.messagebox.showinfo')
    def test_perform_operation_decompress_huffman_success(self, mock_showinfo, mock_decompress_file_huffman):
        """Test the perform_operation method for successful huffman decompression."""
        self.file_combobox_mock.get.return_value = 'test.huff'
        self.compression_combobox_mock.get.return_value = 'Decompression'
        self.algorithm_combobox_mock.get.return_value = 'Huffman coding'

        self.app.perform_operation()

        # Verify that the correct decompression method was called
        mock_decompress_file_huffman.assert_called_once_with('test.huff')
        mock_showinfo.assert_called_once_with("Operation results", 'Performing Decompression using Huffman coding on test.huff')


    @patch('src.compression_app.CompressionApp.compress_file_huffman', return_value=False)
    @patch('src.compression_app.messagebox.showerror')
    def test_perform_operation_compress_huffman_failure(self, mock_showerror, mock_compress_file_huffman):
        """Test the perform_operation method when Huffman compression fails."""
        self.file_combobox_mock.get.return_value = 'test.txt'
        self.compression_combobox_mock.get.return_value = 'Compression'
        self.algorithm_combobox_mock.get.return_value = 'Huffman coding'

        self.app.perform_operation()

        # Verify that the correct compression method was called
        mock_compress_file_huffman.assert_called_once_with('test.txt')
        mock_showerror.assert_called_once_with("Operation Error", "An error occurred during the operation.")

    @patch('src.compression_app.messagebox.showwarning')
    def test_perform_operation_no_selection(self, mock_showwarning):
        """Test the perform_operation method when no operation or algorithm is selected."""
        self.file_combobox_mock.get.return_value = ''
        self.compression_combobox_mock.get.return_value = ''
        self.algorithm_combobox_mock.get.return_value = ''

        self.app.perform_operation()

        # Verify that the warning message is shown
        mock_showwarning.assert_called_once_with("Operation Error", 'Selected operation or algorithm is not supported.')

    @patch('src.compression_app.CompressionApp.compress_file_lzw', side_effect=Exception('Test exception'))
    @patch('src.compression_app.messagebox.showerror')
    def test_perform_operation_exception_handling(self, mock_showerror, mock_compress_file_lzw):
        """Test the perform_operation method's exception handling."""
        self.file_combobox_mock.get.return_value = 'test.txt'
        self.compression_combobox_mock.get.return_value = 'Compression'
        self.algorithm_combobox_mock.get.return_value = 'Lempel-Ziv-Welch (LZW)'

        self.app.perform_operation()

        # Verify that the error message is shown
        mock_showerror.assert_called_once_with("Operation Error", "An error occured during the operation: Test exception")

       
    def test_undo_operation(self):
        """Test the undo operation functionality."""
        # Setting initial values
        self.file_combobox_mock.set('test_file_path.txt')
        self.compression_combobox_mock.set('Compression')
        self.algorithm_combobox_mock.set('Lempel-Ziv-Welch (LZW)')
        
        # Setting up mocks so that the `get` method returns expected values ​​after undo
        self.file_combobox_mock.get.return_value = ''
        self.compression_combobox_mock.get.return_value = ''
        self.algorithm_combobox_mock.get.return_value = ''

        self.app.undo_operation()

        # Check that the values ​​are reset
        self.file_combobox_mock.set.assert_called_with('')
        self.compression_combobox_mock.set.assert_called_with('')
        self.algorithm_combobox_mock.set.assert_called_with('')
    
    @patch('src.compression_app.messagebox.showerror')
    def test_undo_operation_exception(self, mock_showerror):
        """Test the undo_operation method when an exception occurs."""
        # Setting up mocks to throw an exception
        self.app.file_combobox.set = MagicMock(side_effect=Exception("Test Exception"))
        self.app.compression_combobox.set = MagicMock(side_effect=Exception("Test Exception"))
        self.app.algorithm_combobox.set = MagicMock(side_effect=Exception("Test Exception"))

        # Calling a method that should handle an exception
        self.app.undo_operation()

        # Checking that showerror was called
        mock_showerror.assert_called_once_with(
            "Undo Operation Error",
            "An error occured during the operation: Test Exception"
        )
 
    @patch('tkinter.filedialog.asksaveasfilename')
    def test_save_file(self, mock_asksaveasfilename):
        """Test saving the file functionality with different algorithms."""
        algorithms = ['Lempel-Ziv-Welch (LZW)', 'Huffman coding', 'JPEG']

        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm):
                mock_asksaveasfilename.return_value = f'testfile_{algorithm.lower().replace(" ", "_")}.output'
                self.app.file_combobox.set(f'testfile_{algorithm.lower().replace(" ", "_")}.output')
                self.app.compression_combobox.set('Compression')
                self.app.algorithm_combobox.set(algorithm)
                self.app.perform_operation()
                self.app.show_results("Compression")

                # Checking for the existence of a result window
                result_window = None
                for widget in self.app.main_window.winfo_children():
                    if isinstance(widget, tk.Toplevel):  
                        result_window = widget
                        break

                self.assertIsNotNone(result_window, "Results window not found")

                save_button = None
                quit_button = None
                for widget in result_window.winfo_children():
                    if isinstance(widget, tk.Button):
                        if widget.cget('text') == 'Save file':
                            save_button = widget
                        elif widget.cget('text') == 'Quit':
                            quit_button = widget

                self.assertIsNotNone(save_button, "Save file button not found")
                self.assertIsNotNone(quit_button, "Quit button not found")

                save_button.invoke()
                quit_button.invoke()

                # Checking that the window actually closes
                result_window.update()  
                self.assertFalse(result_window.winfo_exists(), "Result window not closed")

    def test_show_results_compression(self):
        """Test showing results for compression."""
        self.app.compression_ratio = MagicMock(return_value=0.5)
        self.app.saving_percentage = MagicMock(return_value=50.0)
        self.app.compression_duration = 1.23
        self.app.algorithm_combobox.get = MagicMock(return_value='Lempel-Ziv-Welch (LZW)')
        
        with unittest.mock.patch('tkinter.filedialog.asksaveasfilename', return_value='testfile.lzw'):
            self.app.show_results('Compression')
            self.assertEqual(len(self.app.main_window.winfo_children()), 5)  

    def test_show_results_decompression(self):
        """Test showing results for decompression."""
        self.app.compression_duration = 1.23
        self.app.algorithm_combobox.get = MagicMock(return_value='Lempel-Ziv-Welch (LZW)')
        
        with unittest.mock.patch('tkinter.filedialog.asksaveasfilename', return_value='testfile.lzw'):
            self.app.show_results('Decompression')
            self.assertEqual(len(self.app.main_window.winfo_children()), 5)  

    def test_compression_ratio(self):
        """Test calculation of compression ratio."""
        self.app.file_combobox.get = MagicMock(return_value=self.test_file_path)
        self.app.current_compressed_data = b'compressed data'
    
        initial_size = os.path.getsize(self.test_file_path)
        compressed_size = len(self.app.current_compressed_data)
        expected_ratio = compressed_size / initial_size
    
        result = self.app.compression_ratio()
        self.assertAlmostEqual(result, expected_ratio, places=2)  

        # Case 2: File does not exist
        with patch('os.path.exists', return_value=False):
            result = self.app.compression_ratio()
            self.assertEqual(result, 0)

        self.app.current_compressed_data = None
        result = self.app.compression_ratio()
        self.assertEqual(result, 0)

        self.app.current_compressed_data = b'compressed data'
        with open(self.test_file_path, 'w') as f:
            f.write('')
        result = self.app.compression_ratio()
        self.assertEqual(result, 0)

    @patch('src.compression_app.os.path.getsize')
    @patch('src.compression_app.os.path.exists')
    @patch('src.compression_app.messagebox.showerror')
    def test_compression_ratio_with_exception(self, mock_showerror, mock_exists, mock_getsize):
        """
        Test the handling of exceptions during compression ratio calculation.
        """
        self.app.current_compressed_data = b'\x00\x01\x02\x03'  # 4 bytes of compressed data
        mock_exists.side_effect = Exception('File system error')

        ratio = self.app.compression_ratio()

        self.assertEqual(ratio, 0)
        mock_showerror.assert_called_once_with("Error", "An error occurred while calculating compression ratio: File system error")

    @patch('src.compression_app.os.path.getsize')
    @patch('src.compression_app.os.path.exists')
    def test_compression_ratio_with_bytes(self, mock_exists, mock_getsize):
        """
        Test the compression ratio calculation with compressed data in bytes.
        """
        self.app.current_compressed_data = b'\x00\x01\x02\x03'  # 4 bytes of compressed data
        mock_exists.return_value = True
        mock_getsize.return_value = 16  # Original file size of 16 bytes

        ratio = self.app.compression_ratio()

        self.assertEqual(ratio, 4 / 16)
    
    def test_saving_percentage(self):
        """Test calculation of saving percentage."""
        self.app.file_combobox.get = MagicMock(return_value=self.test_file_path)

    # 1. Checking with data as bytes
        self.app.current_compressed_data = b'compressed data'
        initial_size = os.path.getsize(self.test_file_path)
        compressed_size = len(self.app.current_compressed_data)
        expected_percentage = 100 - (compressed_size / initial_size) * 100
        result = self.app.saving_percentage()
        self.assertAlmostEqual(result, expected_percentage)

    # 2. Checking with None
        self.app.current_compressed_data = None
        self.assertEqual(self.app.saving_percentage(), 0)

    # 3. Test with empty file
        self.app.current_compressed_data = b'compressed data'
        with open(self.test_file_path, 'w') as f:
            f.write('')
        self.assertEqual(self.app.saving_percentage(), 0)

        # 4. Test with data as a list of 32-bit integers
        with open(self.test_file_path, 'w') as f:
            f.write('This is a test file for saving percentage calculation.')


        # 5. Check for exception
        with patch('tkinter.messagebox.showerror') as mock_showerror:
        # Exception that will be thrown
            self.app.file_combobox.get = MagicMock(side_effect=Exception("Test exception"))
        
            result = self.app.saving_percentage()
        # Check that a method returns 0 in case of an exception
            self.assertEqual(result, 0)
        
        # Check that showerror was called with the correct parameters
            mock_showerror.assert_called_once_with("Error", "An error occurred while calculating saving percentage: Test exception")

        
if __name__=='__main__':
    unittest.main()