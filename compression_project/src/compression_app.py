import tkinter as tk
import cv2
from tkinter import ttk, filedialog, messagebox
import os
import json
import time
import numpy as np
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/algorithms')))

from algorithms.lzw import LZW  
from algorithms.huffman import Huffman
from file_handler import FileHandler
from algorithms.jpeg import JPEG


class CompressionApp:
    """ 
    Main application class for the compression/decompression tool.
    
    This class sets up the GUI for the application, allowing the user to select files,
    choose an operation (compression or decompression) and select an algorithm.
    """
    def __init__(self):
        """ 
        Initializes the main window and configures the user interface elements
        
        Sets up the main window of the application and organizes the layout
        using frames, labels, combo boxes, and buttons.
        """
        self.main_window = tk.Tk()
        self.main_window.title("Compression Application")
        self.main_window.geometry('600x400')

        # Frames for organizing the layout of the window
        self.top_frame = tk.Frame(self.main_window)
        self.mid_frame_1 = tk.Frame(self.main_window)
        self.mid_frame_2 = tk.Frame(self.main_window)
        self.bottom_frame = tk.Frame(self.main_window)
        
        # Labels and combo boxes for file selection, operation selection and algorithm selection
        self.label_file = tk.Label(self.top_frame, text='File selection')
        self.file_combobox = ttk.Combobox(self.top_frame, state='readonly', width=50)
        self.file_combobox['values'] = []
        self.label_compression = tk.Label(self.mid_frame_1, text='Choose operation')
        self.compression_combobox = ttk.Combobox(self.mid_frame_1, state='readonly', width=30)
        self.label_algorithm = tk.Label(self.mid_frame_2, text='Choose algorithm')
        self.algorithm_combobox = ttk.Combobox(self.mid_frame_2, state='readonly', width=30)

        # Positioning the labels and combo boxes 
        self.label_file.pack(side='left', anchor='w', padx=10, pady=(50, 5))
        self.file_combobox.pack(side='left', padx=10, pady=(50, 5))
        self.label_compression.pack(side='left', anchor='w', padx=10, pady=5)
        self.compression_combobox.pack(side='left', padx=10, pady=5)
        self.label_algorithm.pack(side='left', anchor='w', padx=10, pady=5)
        self.algorithm_combobox.pack(side='left', padx=10, pady=5)

        #Buttons for browsing files, performing operations, undoing operations and exiting the application 
        self.file_button = tk.Button(self.top_frame, text='Browse files', command=self.browse_path)
        self.ok_button = tk.Button(self.bottom_frame, text='OK', command=self.perform_operation)
        self.undo_button = tk.Button(self.bottom_frame, text='Undo Operation', command=self.undo_operation)
        self.quit_button = tk.Button(self.bottom_frame, text='Quit', command=self.main_window.destroy)
        
        # Positioning the buttons
        self.file_button.pack(side='left', padx=10, pady=(50, 5))
        self.ok_button.pack(side='left', padx=10, pady=5)
        self.undo_button.pack(side='left', padx=10, pady=5)
        self.quit_button.pack(side='left', padx=10, pady=5)

        # Packing frames
        self.top_frame.pack(anchor='w', fill='x', padx=10, pady=(50, 5))
        self.mid_frame_1.pack(anchor='w', fill='x', padx=10, pady=5)
        self.mid_frame_2.pack(anchor='w', fill='x', padx=10, pady=5)
        self.bottom_frame.pack(anchor='w', fill='x', padx=10, pady=5)
        
        # Variables to store current compressed data and codebook
        self.current_compressed_data = None
        self.current_codebook = None
       
        self.lzw = LZW()
        self.huffman = Huffman()
        self.jpeg = JPEG()
        self.filehandler = FileHandler()

        # Populate combo boxes with choices
        self.choice_operation()
        self.choice_algorithm()

    def choice_operation(self):
        """
        Populates the operation selection combo box with options: 'Compression', 'Decompression'

        This method sets the possible values for the combo box widget that allows users
        to select between the compression and decompression operations.
        """
        operations = ['Compression', 'Decompression']
        self.compression_combobox['values'] = operations

    def browse_path(self):
        """
        Opens a file dialog to select a file and populates the file selection combo box with the selected file
        
        This method allows the user to browse the filesystem and select a file, which is then
        displayed in the file selection combo box.
        """
        file_path = filedialog.askopenfilename()
        if file_path:
            current_files = list(self.file_combobox['values'])
            current_files.append(file_path)
            self.file_combobox['values'] = current_files
            self.file_combobox.set(file_path)

    def choice_algorithm(self):
        """
        Populates the algorithm selection combo box with options: 'Lempel-Ziv-Welch (LZW)', 'Huffman coding', 'JPEG'
        
        This method sets the possible values for the combo box widget that allows users
        to select the compression or decompression algorithm they wish to use.
        """
        algorithms = ['Lempel-Ziv-Welch (LZW)', 'Huffman coding', 'JPEG']
        self.algorithm_combobox['values'] = algorithms

    def perform_operation(self):
        """
        Determines the selected operation and algorithm, and performs the corresponding compression or decompression
        
        Based on the user's selection, this method performs the specified operation (compression or decompression)
        using the chosen algorithm (LZW, Huffman, or JPEG). It then displays the result or an error message if
        something goes wrong.
        
        Exceptions:
            Displays an error message if the operation fails for any reason.
        """
        try:
            selected_file = self.file_combobox.get()
            selected_operation = self.compression_combobox.get()
            selected_algorithm = self.algorithm_combobox.get()
            print(selected_file)
            if selected_file and selected_operation and selected_algorithm:
                start_time = time.time() # Timer starts

                success = False

                if selected_operation == 'Compression':
                    if selected_algorithm == 'Lempel-Ziv-Welch (LZW)':
                        success = self.compress_file_lzw(selected_file)
                    elif selected_algorithm == 'Huffman coding':
                        success = self.compress_file_huffman(selected_file)
                    elif selected_algorithm == 'JPEG':
                        success = self.compress_file_jpeg(selected_file)
                elif selected_operation == 'Decompression':
                    if selected_algorithm == 'Lempel-Ziv-Welch (LZW)':
                        success = self.decompress_file_lzw(selected_file)
                    elif selected_algorithm == 'Huffman coding':
                        success = self.decompress_file_huffman(selected_file)
                    elif selected_algorithm == 'JPEG':
                        success = self.decompress_file_jpeg(selected_file)
                        
                end_time = time.time()
                self.compression_duration = end_time-start_time
                
                if success:
                    result_message = f"Performing {selected_operation} using {selected_algorithm} on {selected_file}"
                    messagebox.showinfo("Operation results", result_message)
                    self.show_results(selected_operation)
                else:
                    messagebox.showerror("Operation Error", "An error occurred during the operation.")
            else:
                messagebox.showwarning("Operation Error", 'Selected operation or algorithm is not supported.')
        except Exception as e:
            messagebox.showerror("Operation Error", f"An error occured during the operation: {str(e)}")

    def undo_operation(self):
        """
        Resets the selected file, operation, and algorithm in the UI
        
        This method clears the selections in the file, operation, and algorithm combo boxes,
        effectively resetting the user interface to its initial state. If an error occurs during
        the reset process, an error message is displayed to the user.
        
        Exceptions:
            Displays an error message if the reset operation fails for any reason.
        """
        try:
            # Reset the selected file, operation, and algorithm
            self.file_combobox.set('')
            self.compression_combobox.set('')
            self.algorithm_combobox.set('')
        except Exception as e:
            # Display an error message if an exception occurs
            messagebox.showerror("Undo Operation Error", f"An error occured during the operation: {str(e)}")
                                      
    def show_results(self, operation):
        """
        Displays a window with the results of the compression or decompression (compression ratio, saving percentage and compression time)
        
        This method creates a new window to display the results of the operation performed. If the
        operation is 'Compression', it shows the compression ratio, saving percentage, and the time taken.
        If the operation is 'Decompression', it shows the time taken for decompression.

        Parameters:
            operation (str): The type of operation performed ('Compression' or 'Decompression').

        Raises:
            Exception: If an error occurs while creating or displaying the results window.
  
        """
        try: 
            # Create a new top-level window to display results
            result_window = tk.Toplevel(self.main_window)
            result_window.geometry('600x400')
            result_window.title('File processing results')
    
            if operation == "Compression":
                # Calculate and display compression results
                compression_ratio = self.compression_ratio()
                saving_percentage = self.saving_percentage()
                
                label_compression_ratio = tk.Label(result_window, text=f"Compression ratio {compression_ratio:.2f}")
                label_saving_percentage = tk.Label(result_window, text=f"Saving percentage {saving_percentage:.2f}%")
                label_compression_time = tk.Label(result_window, text=f"Compression time {self.compression_duration:.2f} seconds")
                
                label_compression_ratio.pack(anchor='w', padx=10, pady=(50, 5))
                label_saving_percentage.pack(anchor='w', padx=10, pady=5)
                label_compression_time.pack(anchor='w', padx=10, pady=5)

                # Create a 'Save file' button with a command based on the selected algorithm
                if self.algorithm_combobox.get() == 'Lempel-Ziv-Welch (LZW)':
                    save_button = tk.Button(result_window, text='Save file', command=self.save_compressed_file_lzw)
                elif self.algorithm_combobox.get() == 'Huffman coding':
                    save_button = tk.Button(result_window, text='Save file', command=self.save_compressed_file_huffman)
                elif self.algorithm_combobox.get() == 'JPEG':
                    save_button = tk.Button(result_window, text='Save file', command=self.save_compressed_file_jpeg)
                else:
                    save_button = tk.Button(result_window, text='Save file', state='disabled')

                
            elif operation == "Decompression":
                # Display decompression time
                label_decompression_time = tk.Label(result_window, text=f"Decompression time {self.compression_duration:.2f} seconds")    
                label_decompression_time.pack(anchor='w', padx = 10, pady =(50, 5))
                
                # Create a 'Save file' button with a command based on the selected algorithm
                if self.algorithm_combobox.get() == 'Lempel-Ziv-Welch (LZW)':
                    save_button = tk.Button(result_window, text='Save file', command= self.save_decompressed_file_lzw)
                elif self.algorithm_combobox.get() == 'Huffman coding':
                    save_button = tk.Button(result_window, text='Save file', command=self.save_decompressed_file_huffman)
                elif self.algorithm_combobox.get() == 'JPEG':
                    save_button = tk.Button(result_window, text='Save file', command=self.save_decompressed_file_jpeg)               
                else:
                    save_button = tk.Button(result_window, text='Save file', state='disabled')

            # Create a 'Quit' button to close the results window
            quit_button = tk.Button(result_window, text='Quit', command=result_window.destroy)
            save_button.pack(side='left', padx=10, pady=5)
            quit_button.pack(side='left', padx=10, pady=5)

        except Exception as e:
            # Show an error message if something goes wrong
            messagebox.showerror("Results error", f"An error occured while showing results: {str(e)}")

    def compression_ratio(self):
        """
        Calculates the compression ratio based on the size of the original file and the size of compressed data
        
        The compression ratio is calculated as the ratio of the size of the compressed data to the size of the original file.
        If the size of the original file is zero or the compressed data is not available, the ratio is returned as 0.

        Returns:
            float: The compression ratio. A value of 0 indicates that compression was not possible or that the size is zero.

        Raises:
            Exception: If an error occurs during the calculation of the compression ratio.
        """
        try:
            # Check if compressed data is available
            if self.current_compressed_data is None:
                return 0

            # Get the path of the selected file
            selected_file = self.file_combobox.get()
            print(selected_file)
            # Check if the selected file exists
            if not os.path.exists(selected_file):
                return 0

            # Get the size of the original file
            initial_size = os.path.getsize(selected_file)
            print("Initial file size is", initial_size)
            # Return 0 if the original file size is zero
            if initial_size == 0:
                return 0
            
            # Calculate the size of the compressed data
            if isinstance(self.current_compressed_data, list):
                # Each list element is a 16-bit integer (2 bytes)
                compressed_size = len(self.current_compressed_data) * 2 
            elif isinstance(self.current_compressed_data, (bytes, bytearray)):
                compressed_size = len(self.current_compressed_data)
            else:
                # For other data types, assume a default size calculation
                compressed_size = len(self.current_compressed_data)
            print("compressed size", compressed_size)         
            # Return the compression ratio
            return compressed_size / initial_size
    
        except Exception as e:
            # Show an error message if something goes wrong
            messagebox.showerror("Error", f"An error occurred while calculating compression ratio: {str(e)}")
            return 0

    def saving_percentage(self):
       """
       Calculates the saving percentagebased on the compression ratio.

       It is calculated as `100 - (compressed_size / original_size) * 100`. 
       The saving percentage is computed as the percentage reduction in file size after compression.
       If the original file size is zero or the compressed data is not available, the saving percentage is returned as 0.

       Returns:
            float: The saving percentage. A value of 0 indicates that no savings were possible or that the size is zero.

       Raises:
            Exception: If an error occurs during the calculation of the saving percentage.
       """
       try:
             # Check if compressed data is available
            if self.current_compressed_data is None:
                return 0
            
            # Get the size of the original file
            initial_size = os.path.getsize(self.file_combobox.get())

            # Return 0 if the original file size is zero
            if initial_size == 0:
                return 0
            
            # Calculate the size of the compressed data
            if isinstance(self.current_compressed_data, list):
                # Each list element is a 16-bit integer (2 bytes)
                compressed_size = len(self.current_compressed_data) * 2
            elif isinstance(self.current_compressed_data, (bytes, bytearray)):
                compressed_size = len(self.current_compressed_data)
            else:
                # For other data types, assume a default size calculation
                compressed_size = len(self.current_compressed_data)    

            
            # Calculate and return the saving percentage
            return max(0, 100 - (compressed_size / initial_size) * 100)
       except Exception as e:
            # Show an error message if something goes wrong
            messagebox.showerror("Error", f"An error occurred while calculating saving percentage: {str(e)}")
            return 0
       
    def save_compressed_file_lzw(self):
        """ 
        Saves the compressed file and corresponding codebook to disk in LZW format.

        This method opens a file dialog for the user to specify the save location
        and then writes the compressed data, codebook, and original file extension 
        to separate files. The compressed data is saved as a `.lzw` file, the 
        codebook is saved as a `.codebook` file, and the original file extension 
        is saved as a `.ext` file.
    
        The compressed data is saved as 4-byte integers, while the codebook entries 
        are saved as the length of the key (1 byte), the key itself, and the value 
        (4-byte integer). The original file extension is saved as a text file.
        """
        if self.current_compressed_data and self.current_codebook:
            # Open file dialog to get the path for saving the compressed data
            file_path = filedialog.asksaveasfilename(defaultextension='.lzw', filetypes=[('LZW files', '*.lzw')])
            if file_path:
                try:
                    # Save compressed data to the specified file
                    with open(file_path, 'wb') as file:
                        for code in self.current_compressed_data:
                            file.write(code.to_bytes(2, byteorder='big'))
                    
                    # Save codebook to a separate file
                    codebook_file_path = file_path + '.codebook'
                    with open(codebook_file_path, 'wb') as file:
                        for key, value in self.current_codebook.items():
                            key_bytes = key  # Convert hex string to bytes
                            value_bytes = value.to_bytes(4, byteorder='big')  # Convert value to 4-byte integer
                            file.write(len(key_bytes).to_bytes(1, byteorder='big'))  # Write length of key (1 byte)
                            file.write(key_bytes)  # Write the key itself
                            file.write(value_bytes)  # Write the value

                    # Save the original file extension to a separate text file
                    extension_file_path = file_path + '.ext'
                    with open(extension_file_path, 'w') as file:
                        file.write(self.original_extension)

                    messagebox.showinfo('Success' ,f'Compressed file and codebook saved successfully.')
                except Exception as e:
                    messagebox.showerror('Error', f'Failed to save files: {str(e)}')
            else:
                messagebox.showwarning('Warning', f"No file path selected")
        else:
            messagebox.showerror('Error', f'No compressed data to save')

    def save_decompressed_file_lzw(self):
        """ 
        Saves the decompressed file to disk
        
        This method opens a file dialog for the user to specify the save location
        and then writes the decompressed data to the selected file. The file extension 
        used is based on the original extension saved during compression.
    
        The method handles errors by displaying an appropriate message box and checks 
        if a file path has been selected before attempting to save.
        """
                
        # Open file dialog to get the path for saving the decompressed data
        file_path = filedialog.asksaveasfilename(
            defaultextension=self.original_extension, 
            filetypes=[(f'{self.original_extension} files', f'*{self.original_extension}')]
            )
        
        if file_path:
            try:
                # Save decompressed data to the specified file
                with open(file_path, 'wb') as file:
                    file.write(self.decompressed_data)
                messagebox.showinfo("Success", "Decompressed file successfully saved")
            except Exception as e:
                # Display an error message if something goes wrong
                messagebox.showerror("Error", f'Failed to save file: {str(e)}')
        else:
            # Display a warning if no file path was selected
            messagebox.showwarning("Warning", "File path not selected")

    def compress_file_lzw(self, file_path):
        """ 
        Performs file compression using the LZW algorithm.

        This method reads the file data from the given file path, performs LZW compression,
        and stores the compressed data and codebook. It also saves the original file extension
        for later use. After compression, it displays a message indicating the success of the operation.

        Parameters:
            file_path (str): The path of the file to be compressed.
    
        Raises:
            Exception: If any error occurs during file reading, compression, or processing.
        """
        try:
            # Read the file data
            data = FileHandler.read_file(file_path, mode='rb')
            
            # Save the original file extension
            self.original_extension = os.path.splitext(file_path)[1]  # Includes the dot

            # Initialize and use the LZW compressor
            lzw = LZW()
            compressed_data, codebook = lzw.compress(data)

            # Store the compressed data and codebook
            self.current_compressed_data = compressed_data
            self.current_codebook = codebook

            # Notify the user of successful compression
            messagebox.showinfo('Compression results', f'File successfully compressed')
            return True

        except Exception as e:
            # Display an error message if something goes wrong
            messagebox.showinfo('Compression Error', f"Failed to compress file: {str(e)}")    
            return False

    def decompress_file_lzw(self, file_path):
        """ 
        Performs decompression of a file compressed using the LZW algorithm.

        This method reads the compressed data and codebook from the given file path, performs LZW decompression,
        and saves the decompressed data to a new file. It also retrieves the original file extension from a separate
        file to properly name the decompressed output file. After decompression, it displays a message indicating
        the success of the operation.

        Parameters:
            file_path (str): The path of the compressed file (with .lzw extension).

        Raises:
            Exception: If any error occurs during file reading, decompression, or processing.
        """     
        try:
            # Check if the file has the correct extension
            if not file_path.endswith('.lzw'):
                messagebox.showerror("Decompression Error", "Selected file does not have the .lzw extension.")
                return False
            
            # Define file paths
            compressed_file_path = file_path 
            codebook_file_path = compressed_file_path + '.codebook'
            extension_file_path = compressed_file_path + '.ext'

            # Check if the required files exist
            if not os.path.exists(compressed_file_path) or not os.path.exists(codebook_file_path):
                messagebox.showerror("Decompression Error", "Compressed file or codebook not found.") 
                return False
        
            # Load compressed data
            compressed_data = []
            with open(compressed_file_path, 'rb') as f:
                while True:
                    bytes_read = f.read(2)
                    if not bytes_read:
                        break
                    compressed_data.append(int.from_bytes(bytes_read, byteorder='big'))

            # Load codebook    
            codebook = FileHandler.load_codebook_lzw(codebook_file_path)
                    
            # Load original extension
            with open(extension_file_path, 'r') as file:
                original_extension = file.read().strip()

            # Decompress the data
            lzw = LZW()
            
            decompressed_data = lzw.decompress(compressed_data, codebook)
            
            original_extension = original_extension
            self.decompressed_data = decompressed_data
            self.original_extension = original_extension
                   
           # Notify user of successful decompression
            messagebox.showinfo('Decompression results', f'File successfully decompressed')
            return True

        except Exception as e:
            # Show an error message if something goes wrong
            messagebox.showerror("Decompression Error", f"Error during decompression: {str(e)}")     
            return False
        
    def save_compressed_file_huffman(self):
        """ 
        Saves the compressed file and corresponding codebook to disk using Huffman encoding.
    
        This method writes the compressed data to a file and saves the Huffman codebook and original file extension 
        in separate files. The codebook is serialized with keys and values converted to bytes.
        """
        def bits_to_bytes(bits):
            """ Converts a binary string to bytes with padding. """
            bits_length = len(bits)
            padded_bits = bits.ljust((bits_length + 7) // 8 * 8, '0')
            padding_length = len(padded_bits) - bits_length
            byte_array = bytearray()
            for i in range(0, len(padded_bits), 8):
                byte = padded_bits[i:i+8]
                byte_array.append(int(byte, 2))
            return bytes(byte_array), padding_length
               
        if self.current_compressed_data and self.current_codebook:
            file_path = filedialog.asksaveasfilename(defaultextension='.huff', filetypes=[('Huffman files', '*.huff')])
            if file_path:
                try:
                    # Save compressed file as binary string
                    with open(file_path, 'wb') as file:
                        file.write(self.current_compressed_data)
                    
                    #Save codebook
                    codebook_file_path = file_path + '.codebook'
                    with open(codebook_file_path, 'wb') as file:
                        for key, value in self.current_codebook.items():
                        # Save key (int) as bytes
                            key_bytes = key.to_bytes((key.bit_length() + 7) // 8 or 1, byteorder='big')
                            file.write(len(key_bytes).to_bytes(1, byteorder='big'))
                            file.write(key_bytes)

                            # Convert value (bit string) to bytes
                            value_bytes, padding_length = bits_to_bytes(value)
                            file.write(len(value_bytes).to_bytes(1, byteorder='big'))
                            file.write(value_bytes)
                            file.write(padding_length.to_bytes(1, byteorder='big'))

                            
                    #Save original file extension
                    extension_file_path = file_path + '.ext'
                    with open(extension_file_path, 'w') as file:
                        file.write(self.original_extension)

                    messagebox.showinfo("Compression results", f'Compressed file and codebook saved successfully.')
                except Exception as e:
                    messagebox.showerror('Error', f'Failed to save files: {str(e)}')
            else:
                messagebox.showwarning('Warning', f"No file path selected")
        else:
            messagebox.showerror('Error', f'No compressed data to save')

    def save_decompressed_file_huffman(self):
        """ 
        Saves the decompressed file to disk.

        This method writes the decompressed data to a file using the original file extension 
        saved during the decompression process. It prompts the user to select the file path 
        where the data will be saved.
        """
        if not self.decompressed_data:
            messagebox.showerror("Error"), f"No decompressed data to save."

        # Prompt user to select a file path
        file_path = filedialog.asksaveasfilename(
            defaultextension=self.original_extension, 
            filetypes=[(f'{self.original_extension} files', f'*{self.original_extension}')]
            )
        
        if file_path:
            try:
                # Save decompressed data to the selected file
                with open(file_path, 'wb') as file:
                    file.write(self.decompressed_data)

                # Notify user of successful save
                messagebox.showinfo("Success", "Decompressed file successfully saved")
            except Exception as e:
                # Handle exceptions and show error message
                messagebox.showerror("Error", f'Failed to save file: {str(e)}')
        else:
            # Notify user if no file path was selected
            messagebox.showwarning("Warning", "File path not selected")
            
    def compress_file_huffman(self, file_path):
        """ 
        Performs file compression using the Huffman coding algorithm.

        This method reads the file from the given path, compresses the data using Huffman coding,
        and stores the compressed data and Huffman table. It also saves the original file extension
        for later use.

        Parameters:
            file_path (str): The path of the file to be compressed.

        Raises:
            Exception: If any error occurs during file reading or compression.
        """
        try:            
            # Read file data
            data = FileHandler.read_file(file_path, mode='rb')
        
            if not data:
                messagebox.showwarning("Compression Warning", "The file is empty or could not be read.")
                return
            
            # Save the original file extension
            self.original_extension = os.path.splitext(file_path)[1]  # Includes the dot

            # Perform Huffman compression
            huffman = Huffman()
            compressed_data, huffman_table = huffman.compress(data)

            # Store the compressed data and Huffman table
            self.current_compressed_data = compressed_data
            self.current_codebook = huffman_table

            # Notify the user of successful compression
            messagebox.showinfo('Compression results', f'File successfully compressed')
            return True
        
        except Exception as e:
            # Handle exceptions and show an error message
            messagebox.showerror('Compression Error', f'An error occurred during compression: {str(e)}')
            return False
        
    def decompress_file_huffman(self, file_path):
        """ 
        Performs decompression of a file compressed using the Huffman algorithm
        """     
        try:
            # Check if the file has the correct extension
            if not file_path.endswith('.huff'):
                messagebox.showerror("Decompression error", "Selected file does not have the .huff extension.")
                return False
        
        # Define paths for the codebook and original extension files
            compressed_file_path = file_path 
            codebook_file_path = compressed_file_path + '.codebook'
            extension_file_path = compressed_file_path + '.ext'
        
            # Check if the required files exist
            if not os.path.exists(compressed_file_path) or not os.path.exists(codebook_file_path):
                messagebox.showerror("Decompression error", "Compressed file or codebook not found.") 
                return False

            # Load the compressed data from the file
            with open(compressed_file_path, 'rb') as f:
                compressed_data = f.read()
  
            # Load the Huffman codebook    
            codebook = FileHandler.load_codebook_huffman(codebook_file_path)
                  
            # Load the original file extension
            with open(extension_file_path, 'r') as file:
                original_extension = file.read().strip()
        
            # Decompress the data using the Huffman algorithm
            huffman = Huffman()
            decompressed_data = huffman.decompress(compressed_data, codebook)
            
            # Store the decompressed data and the original extension
            self.decompressed_data = decompressed_data
            self.original_extension = original_extension
            
            # Notify the user of successful decompression
            messagebox.showinfo('Decompression results', f'File successfully decompressed')
            return True
        
        except Exception as e:
            # Handle exceptions and show an error message
            messagebox.showerror("Decompression error", f"Error during decompression: {str(e)}")
            return False

        
    def compress_file_jpeg(self, file_path):
        """ 
        Performs file compression using the JPEG coding algorithm.

        
        """
        try:            
            # Read file data
            data = FileHandler.read_image_file(file_path)
            #if not data:
               # messagebox.showwarning("Compression Warning", "The file is empty or could not be read.")
                #return
                        
            # Save the original file extension
            self.original_extension = os.path.splitext(file_path)[1]  # Includes the dot

             # Perform JPEG compression
            jpeg = JPEG()
            compressed_huffman, huffman_table, length, height, luminance_length, luminance_height, blue_length, blue_height, red_length, red_height, len_rle_lum, len_rle_blue, len_rle_red = jpeg.compress(data)
             
            self.current_compressed_data = compressed_huffman
            self.current_codebook = huffman_table
            self.length = length
            self.height = height
            self.luminance_length = luminance_length
            self.luminance_height = luminance_height
            self.blue_length = blue_length
            self.blue_height = blue_height
            self.red_length = red_length
            self.red_height = red_height
            self.len_rle_lum = len_rle_lum
            self.len_rle_blue = len_rle_blue  
            self.len_rle_red = len_rle_red
            
            # Notify the user of successful compression
            messagebox.showinfo('Compression results', f'File successfully compressed')
            return True
        
        except Exception as e:
            # Handle exceptions and show an error message
            messagebox.showerror('Compression Error', f'An error occurred during compression: {str(e)}')
            return False
    
    def save_compressed_file_jpeg(self):
        """
    Saves the compressed JPEG file and corresponding codebook to disk using Huffman encoding.

    This method writes the compressed data to a file and saves the Huffman codebook and original file extension
    in separate files. The codebook is serialized with keys and values converted to bytes.
        """
        def bits_to_bytes(bits):
            """ Converts a binary string to bytes with padding. """
            bits_length = len(bits)
            padded_bits = bits.ljust((bits_length + 7) // 8 * 8, '0')
            padding_length = len(padded_bits) - bits_length
            byte_array = bytearray()
            for i in range(0, len(padded_bits), 8):
                byte = padded_bits[i:i+8]
                byte_array.append(int(byte, 2))
            return bytes(byte_array), padding_length

        if self.current_compressed_data and self.current_codebook:
            # Save compressed JPEG file
            jpeg_file_path = filedialog.asksaveasfilename(defaultextension='.huff', filetypes=[('Huffman files', '*.huff')])
            if jpeg_file_path:
                try:
                # Save compressed JPEG data
                    with open(jpeg_file_path, 'wb') as file:
                        file.write(self.current_compressed_data)

                # Save Huffman codebook
                    codebook_file_path = jpeg_file_path + '.codebook'
                    with open(codebook_file_path, 'wb') as file:
                        for key, value in self.current_codebook.items():
                        # Save key (int) as bytes
                            key_bytes = key.to_bytes((key.bit_length() + 7) // 8 or 1, byteorder='big')
                            file.write(len(key_bytes).to_bytes(1, byteorder='big'))
                            file.write(key_bytes)

                        # Convert value (bit string) to bytes
                            value_bytes, padding_length = bits_to_bytes(value)
                            file.write(len(value_bytes).to_bytes(1, byteorder='big'))
                            file.write(value_bytes)
                            file.write(padding_length.to_bytes(1, byteorder='big'))

                    # Save original file extension
                    extension_file_path = jpeg_file_path + '.ext'
                    with open(extension_file_path, 'w') as file:
                        file.write(self.original_extension)
                    
                    # Save additional data to a separate file
                    additional_data_file_path = jpeg_file_path + '.additional'
                    with open(additional_data_file_path, 'wb') as file:
                    # Save lengths and then the actual data
                        for data in [self.length, self.height, self.luminance_length, self.luminance_height, self.blue_length, self.blue_height, self.red_length, self.red_height, self.len_rle_lum, self.len_rle_blue, self.len_rle_red]:
                            length_data = len(data)
                            file.write(length_data.to_bytes(4, byteorder='big'))  # Save the length as 4 bytes
                            file.write(data)

                    messagebox.showinfo("Compression results", f'Compressed JPEG file and codebook saved successfully.')
                except Exception as e:
                    messagebox.showerror('Error', f'Failed to save files: {str(e)}')
            else:
                messagebox.showwarning('Warning', f"No file path selected")
        else:
            messagebox.showerror('Error', f'No compressed data to save')
        
    def decompress_file_jpeg(self, file_path):
        """ 
        Performs JPEG decompression of a file compressed using the Huffman algorithm
        """     
        try:
            # Check if the file has the correct extension
            if not file_path.endswith('.huff'):
                messagebox.showerror("Decompression error", "Selected file does not have the .huff extension.")
                return False
        
        # Define paths for the codebook and original extension files
            compressed_file_path = file_path 
            codebook_file_path = compressed_file_path + '.codebook'
            extension_file_path = compressed_file_path + '.ext'
            additional_data_file_path = compressed_file_path + '.additional'
            #_dict_file_path = compressed_file_path + '.freq'
            # Check if the required files exist
            if not os.path.exists(compressed_file_path) or not os.path.exists(codebook_file_path):
                messagebox.showerror("Decompression error", "Compressed file or codebook not found.") 
                return False

            # Load the compressed data from the file
            with open(compressed_file_path, 'rb') as f:
                compressed_data = f.read()
           
            # Load the Huffman codebook    
            codebook = FileHandler.load_codebook_huffman(codebook_file_path)
                   
            # Load the original file extension
            with open(extension_file_path, 'r') as file:
                original_extension = file.read().strip()
                       
            # Load additional data (image dimensions and channel bytes)
            with open(additional_data_file_path, 'rb') as f:
            # Read length and data for length
                length_data_size = int.from_bytes(f.read(4), byteorder='big')
                length = f.read(length_data_size)

                # Read length and data for height
                height_data_size = int.from_bytes(f.read(4), byteorder='big')
                height = f.read(height_data_size)

                # Read length and data for blue_length
                luminance_length_data_size = int.from_bytes(f.read(4), byteorder='big')
                luminance_length = f.read(luminance_length_data_size)

                # Read length and data for blue_height
                luminance_height_data_size = int.from_bytes(f.read(4), byteorder='big')
                luminance_height = f.read(luminance_height_data_size)

                # Read length and data for blue_length
                blue_length_data_size = int.from_bytes(f.read(4), byteorder='big')
                blue_length = f.read(blue_length_data_size)

                # Read length and data for blue_height
                blue_height_data_size = int.from_bytes(f.read(4), byteorder='big')
                blue_height = f.read(blue_height_data_size)

                # Read length and data for red_length
                red_length_data_size = int.from_bytes(f.read(4), byteorder='big')
                red_length = f.read(red_length_data_size)

                # Read length and data for red_height
                red_height_data_size = int.from_bytes(f.read(4), byteorder='big')
                red_height = f.read(red_height_data_size)

                len_rle_lum_data_size = int.from_bytes(f.read(4), byteorder='big')
                len_rle_lum = f.read(len_rle_lum_data_size)
                
                len_rle_blue_data_size = int.from_bytes(f.read(4), byteorder='big')
                len_rle_blue = f.read(len_rle_blue_data_size)
                
                len_rle_red_data_size = int.from_bytes(f.read(4), byteorder='big')
                len_rle_red = f.read(len_rle_red_data_size)
                
            # Decompress the data using the Huffman algorithm
            jpeg = JPEG()
            decompressed_data = jpeg.decompress(compressed_data, codebook, length, height, luminance_length, luminance_height, blue_length, blue_height, red_length, red_height, len_rle_lum, len_rle_blue, len_rle_red)
            
            # Store the decompressed data and the original extension
            self.decompressed_data = decompressed_data
            self.original_extension = original_extension
            
            # Notify the user of successful decompression
            messagebox.showinfo('Decompression results', f'File successfully decompressed')
            return True
        
        except Exception as e:
            # Handle exceptions and show an error message
            messagebox.showerror("Decompression error", f"Error during decompression: {str(e)}")
            return False
      
   
    def save_decompressed_file_jpeg(self):
        """ 
        Saves the decompressed file to disk.
        """
        
        # Prompt user to select a file path
        file_path = filedialog.asksaveasfilename(
            defaultextension=self.original_extension, 
            filetypes=[(f'{self.original_extension} files', f'*{self.original_extension}')]
            )
        
        if file_path:
            
            try:
                # Save decompressed data to file
                # Assuming decompressed_data is in a format that cv2.imwrite can handle (e.g., numpy array)
                with open(file_path, 'wb') as file:
                    cv2.imwrite(file_path, self.decompressed_data)

            # Notify user of successful save
                messagebox.showinfo("Success", "Image successfully saved as PNG")
            except Exception as e:
            # Handle exceptions and show error message
                messagebox.showerror("Error", f'Failed to save image: {str(e)}')
        else:
        # Notify user if no file path was selected
            messagebox.showwarning("Warning", "File path not selected")
    
    

def run_app():
    app = CompressionApp()
    app.main_window.mainloop()
    

