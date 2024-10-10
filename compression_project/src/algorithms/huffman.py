
import heapq


class Huffman:
    """
    Class for encoding and decoding data using the Huffman coding algorithm.
    
    Huffman coding is a compression algorithm that assigns variable-length codes to input characters,
    with shorter codes assigned to more frequent characters. 

    Attributes:
    root (HTNode): The root node of the Huffman tree.
    """
    class HTNode:
        """
        Node class used for building the Huffman tree.
        
        Each node represents a character or a combination of characters in the Huffman tree. It contains
        the character's frequency, its Huffman code, and references to its left and right children in the tree.
        
        Attributes:
            value (int): The frequency of the character or the combined frequency of the subtree.
            key (str): The character represented by this node (for leaf nodes) or an empty string (for internal nodes).
            binary (str): The Huffman code assigned to this character.
            left (HTNode or None): The left child of this node.
            right (HTNode or None): The right child of this node.
        """
        def __init__(self, value, key):
            """
            Initializes an HTNode instance.
            
            Args:
                value (int): The frequency of the character or the combined frequency of the subtree.
                key (str): The character represented by this node (for leaf nodes) or an empty string (for internal nodes).
            """
            self.key = key
            self.value = value
            self.binary = ""
            self.left = None
            self.right = None

        def add_nodes(self,node1,node2):
            """
        Adds two nodes as left and right children of this node.

        Args:
            node1 (HTNode): The first node to be added.
            node2 (HTNode): The second node to be added.

        Notes:
            - The node with the smaller frequency (`value`) becomes the left child.
        """
            if node1.value <= node2.value:
                self.left = node1
                self.right = node2
            else:
                self.left = node2
                self.right = node1

        def __lt__(self, other):
            return self.value < other.value
        
    def create_parent_node(self, node1, node2):
        """
        Creates a parent node from two child nodes.

        Args:
            node1 (Huffman.HTNode): The first child node.
            node2 (Huffman.HTNode): The second child node.

        Returns:
            Huffman.HTNode: A new parent node combining `node1` and `node2`.

        Notes:
            - The parent node's frequency is the sum of `node1` and `node2` frequencies.
            - The parent node's key is the concatenation of `node1` and `node2` keys.
            - The `add_nodes` method sets `node1` and `node2` as left and right children of the parent node.
        """
        if node1 is None or node2 is None:
            raise ValueError("Cannot create a parent node with NoneType nodes.")
        
        parentNode = self.HTNode(node1.value + node2.value, node1.key + node2.key)
        parentNode.add_nodes(node1,node2)
        return parentNode
    
    def huffman(self, data):
        """
        Compresses input data using the Huffman coding algorithm.

        This method builds a Huffman tree from the input data and generates a Huffman table
        that maps each byte to its corresponding Huffman code. 

        Args:
            data (bytes): The input data to be compressed. It should be in bytes format.

        Returns:
            dict: A dictionary representing the Huffman table. 
            The keys are the bytes from the input data, and the values are their corresponding Huffman codes.
        """
        if not data:
            return {}, {}
        
        # Create a frequency table for each byte in the input data
        huffman_table = {}
    
        for byte in data:
            huffman_table[byte] = huffman_table.get(byte, 0) + 1
        
        # Special case: if there's only one unique byte
        if len(huffman_table) == 1:
            single_byte = list(huffman_table.keys())[0]
            return {single_byte: '0'}
        
        # Build the priority queue with Huffman nodes
        priority_queue = []
        for key, value in huffman_table.items():
            heapq.heappush(priority_queue, (value, self.HTNode(value, key)))
            
        
        # Build the Huffman tree
        while len(priority_queue) > 1:
            value1, left = heapq.heappop(priority_queue)
            value2, right = heapq.heappop(priority_queue)
            parent_node = self.create_parent_node(left, right)
            heapq.heappush(priority_queue, (parent_node.value, parent_node))

        # The remaining node in the priority queue is the root of the Huffman tree  
        root = heapq.heappop(priority_queue)[1]
        root.binary = ""
        huffman_table = {}
        self.set_binary(root, huffman_table)
                
        return huffman_table

    def set_binary(self, tree, huffman_table):
        """
        Recursively assigns binary codes to bytes in the Huffman tree and updates the Huffman table.

        Args:
            tree (HTNode): The root node of the Huffman tree (an instance of HTNode).
            huffman_table (dict): Dictionary to update with byte-to-binary code mappings.

        Returns:
            None: The Huffman table is updated in place.
        """
        if tree.left and tree.right:
            tree.left.binary = tree.binary + "0"
            tree.right.binary = tree.binary + "1"
            self.set_binary(tree.left, huffman_table)
            self.set_binary(tree.right, huffman_table)
        else:
            huffman_table[tree.key] = tree.binary

    def compress(self, data):
        """
        Compresses data using Huffman coding.

        Args:
            data (bytes): Input data to compress.

        Returns:
            tuple: A tuple containing:
                - `bytes`: Compressed data with padding.
                - `dict`: Huffman coding table (byte to binary code).
        """
        if not data:
            return b'\x08', {} # Padding byte with no data and empty huffman table
        

        huffman_table = self.huffman(data)
        
        # Encode the data using the Huffman table
        encoded_data = ''.join(huffman_table[byte] for byte in data)
        
        # Convert binary string to byte array
        byte_array = bytearray()

        # Add padding to make binary string length a multiple of 8
        padding = 8 - len(encoded_data)%8
        if padding !=8:
            encoded_data += "0" * padding

        for i in range(0, len(encoded_data), 8):
            byte_chunk = encoded_data[i:i+8]
            byte_array.append(int(byte_chunk, 2))

        # Add padding info as the first byte
        byte_array.insert(0, padding)
       
        return bytes(byte_array), huffman_table


    def decompress(self, encoded_data, huffman_table):
        """
        Decompresses data compressed with Huffman encoding.

        Args:
            encoded_data (bytes): Compressed data with padding info.
            huffman_table (dict): Huffman table mapping binary codes to byte values.

        Returns:
            bytes: Decompressed data.
        """
        if not encoded_data or len(encoded_data) < 2:
            return b''
        
        if len(encoded_data) < 2:
            raise ValueError("Encoded data is empty or too short.")
        
        if not huffman_table:
            raise ValueError("Huffman table is empty or invalid.")
        
        # Extract padding info
        padding = encoded_data[0]

        # Convert encoded data to binary string
        bit_string = ''.join(f'{byte:08b}' for byte in encoded_data[1:])

        # Remove padding bits
        if padding !=8:
            bit_string = bit_string[:-padding]

        # Create Huffman dictionary for decoding
        huffman_dictionary = {v: k for k, v in huffman_table.items()}
        current_code = ""
        decoded_data = bytearray()

        # Decode the binary string
        for bit in bit_string:
            current_code += bit
            if current_code in huffman_dictionary:
                decoded_data.append(huffman_dictionary[current_code])
                current_code = "" 
          
        return bytes(decoded_data)


    