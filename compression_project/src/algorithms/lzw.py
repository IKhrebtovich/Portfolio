

class LZW:
    """
    Class for compressing and decompressing data using the LZW (Lempel-Ziv-Welch) algorithm.
    
    The LZW algorithm is a dictionary-based compression algorithm that replaces repeated occurrences 
    of strings with shorter codes. It is used in various file formats such as GIF and TIFF.
    """
    
    def compress(self, data):
        """ 
        Compresses input data using the LZW algorithm.
        
        This method takes a sequence of bytes and compresses it using the LZW algorithm. It builds 
        a dictionary of substrings and their corresponding codes. As the input data is processed, 
        it generates a list of codes representing the compressed data. The method also returns the 
        dictionary used for compression, which serves as the codebook for decompression.
        
        Args:
            data (bytes): The input data to be compressed.
            
        Returns:
            tuple: A tuple containing two elements:
                - A list of integers representing the compressed data.
                - A dictionary representing the codebook used for compression.
        Notes:
            - The dictionary is initialized with single-byte strings (0-255).
            - The dictionary grows dynamically up to a maximum size of 4096 entries.
            - The codebook is a dictionary mapping byte sequences to their corresponding codes.
        """
        # Initialize dictionary size and the dictionary itself
        dictionary_size = 256
        lzw_dictionary = {bytes([i]): i for i in range(dictionary_size)}
        max_dict_size = 4096
        
        # Initialize current string
        w = bytes()
        
        # List for storing encoded data
        encoded = []
        
        # Main compression loop 
        for byte in data:
            # Concatenate current string with the new character
            wc = w + bytes([byte])
            if wc in lzw_dictionary:
                w = wc
            else:
                # Store current string as encoded data
                encoded.append(lzw_dictionary[w])
                # Add new string to dictionary
                if len(lzw_dictionary) < max_dict_size:
                    lzw_dictionary[wc] = dictionary_size
                    dictionary_size += 1
                # Reset current string for next iteration
                w = bytes([byte])

        # Add remaining string
        if w:
            encoded.append(lzw_dictionary[w])
        
        # Create the codebook by inverting the dictionary
        codebook = {key: value for key, value in lzw_dictionary.items()}
            
        return encoded, codebook

    def decompress(self, encoded, codebook):
        """ 
        This method takes a list of compressed codes and a codebook (dictionary) used for 
        decompression. It reconstructs the original data from the compressed codes using the provided 
        codebook, which maps codes to their corresponding byte sequences.

        Args:
            encoded (list of int): The list of compressed codes to be decompressed.
            codebook (dict): A dictionary mapping codes to their corresponding byte sequences.
            
        Returns:
            bytes: The decompressed data as a byte sequence.
        """
        if not encoded:
            raise ValueError("Compressed data is empty")     
               
        # Initialize the dictionary for decompression
        lzw_dictionary = {v: k for k, v in codebook.items()}
        dictionary_size = len(lzw_dictionary)
                
        # Initialize the decompression process with the first code
        w = lzw_dictionary.get(encoded.pop(0))
        
        if w is None:
            raise ValueError("Initial index not found in dictionary")
        
        result = list(w)
        
        # Main decoding loop
        for k in encoded:
            if k in lzw_dictionary:
                entry = lzw_dictionary[k]
            elif k == dictionary_size:
                entry = w + w[:1]
            else:
                raise ValueError('Invalid compressed data: {k}')
            result.extend(entry)

            # Add a new entry to the dictionary
            lzw_dictionary[dictionary_size] = w + entry[:1]
            dictionary_size += 1
            w = entry
        
        return bytes(result)