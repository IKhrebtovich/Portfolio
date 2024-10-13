This project documents the development and implementation of the most commonly used compression algorithms: lossless (Lempel-Ziv-Welch, Huffman coding) and lossy (JPEG) in a simple application programming interface (API). It has a user-friendly interface that allows users to select files, apply compression, and view the operation results.

Features
- Compress and decompress files using multiple algorithms (Huffman, LZW, JPEG)
- User-friendly graphical interface
- Supports multiple file types for Huffman and LZW
- Supports ".png" files for JPEG
- Displays compression ratio, saving percentage and operation time

Running a project from source code

1.	Dependencies:
-	Python 3.8 or higher.

2.	Cloning the source code:
The source code for the project is available and can be downloaded from GitHub.

Step1.  Clone the repository:
git clone 

Step 2. Install dependencies (if necessary):
pip install -r requirements.txt

Step 3.  Run the application:
src/main.py

Step 4. Run the tests:
python -m unittest discover tests


Project structure:

- `src/` - Contains the main source code
  - `main.py` - The entry point for the application
  - `algorithms/` - Implements various compression algorithms
    - `lzw.py` - Lempel-Ziv-Welch compression
    - `huffman.py` - Huffman coding compression
    - `jpeg.py` - JPEG compression
  - `compression_app.py` - Main application with GUI interface
  - `file_handler.py` - Handles file operations

- `tests/` - Contains unit tests
  - `test_lzw.py` - Tests for LZW compression
  - `test_huffman.py` - Tests for Huffman coding
  - `test_jpeg.py` - Tests for JPEG compression
  - `test_gui.py` - Tests for the graphical user interface
  - `test_file_handler.py` - Tests for file handling
  - `test_compression_app.py` - Tests for the main application

- `requirements.txt` - List of Python dependencies
- `README.md` - This file

