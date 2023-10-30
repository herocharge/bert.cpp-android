import ctypes
from typing import Union, List
# import numpy as np
import os
import sys
import math

N_THREADS = 6

if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    print("Usage: python3 sample_dylib.py <ggml model path>")
    exit(0)

def argsort(arr):
    return sorted(range(len(arr)), key=lambda i: arr[i])

def dot_product(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimension for dot product")

    result = sum(x * y for x, y in zip(vector1, vector2))
    return result

def euclidean_norm(vector):
    squared_sum = sum(x**2 for x in vector)
    result = math.sqrt(squared_sum)
    # print(vector)
    return result + 0.1


def zeros(shape):
    if not isinstance(shape, tuple) or len(shape) == 0:
        raise ValueError("Shape must be a non-empty tuple")

    result = []
    for _ in range(shape[0]):
        if len(shape) > 1:
            result.append(zeros(shape[1:]))
        else:
            result.append(0.0)
    return result

class BertModel:
    def __init__(self, fname):
        self.lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "libbert.so"))

        self.lib.bert_load_from_file.restype = ctypes.c_void_p
        self.lib.bert_load_from_file.argtypes = [ctypes.c_char_p]

        self.lib.bert_n_embd.restype = ctypes.c_int32
        self.lib.bert_n_embd.argtypes = [ctypes.c_void_p]
        
        self.lib.bert_free.argtypes = [ctypes.c_void_p]

        self.lib.bert_encode_batch.argtypes = [
            ctypes.c_void_p,    # struct bert_ctx * ctx,
            ctypes.c_int32,     # int32_t n_threads,  
            ctypes.c_int32,     # int32_t n_batch_size
            ctypes.c_int32,     # int32_t n_inputs
            ctypes.POINTER(ctypes.c_char_p),                # const char ** texts
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), # float ** embeddings
        ]

        self.ctx = self.lib.bert_load_from_file(fname.encode("utf-8"))
        self.n_embd = self.lib.bert_n_embd(self.ctx)

    def __del__(self):
        self.lib.bert_free(self.ctx)

    def encode(self, sentences: Union[str, List[str]], batch_size: int = 16):
        input_is_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_is_string = True

        n = len(sentences)

        embeddings = zeros((n, self.n_embd))
        c_float_arrays = (ctypes.POINTER(ctypes.c_float) * len(embeddings))()

        # # Create float arrays and assign them to the pointers
        for i, row in enumerate(embeddings):
            c_float_arrays[i] = (ctypes.c_float * len(row))(*row)

        # # Now you have a double pointer to a C-type array of floats
        # c_double_pointer = ctypes.cast(c_float_arrays, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))
        # embeddings_pointers = c_double_pointer
        embeddings_pointers = (ctypes.POINTER(ctypes.c_float) * len(embeddings))(*c_float_arrays)
        # embeddings_pointers = (ctypes.POINTER(ctypes.c_float) * n)(((ctypes.c_float * self.n_embd) * n))
        # print(embeddings_pointers)
        texts = (ctypes.c_char_p * n)()
        for j, sentence in enumerate(sentences):
            texts[j] = sentence.encode("utf-8")

        self.lib.bert_encode_batch(
            self.ctx, N_THREADS, batch_size, len(sentences), texts, embeddings_pointers
        )
        for i in range(n):
            for j in range(self.n_embd):
                embeddings[i][j] = embeddings_pointers[i][j]
        # embeddings = embeddings_pointers()
        if input_is_string:
            return embeddings[0]
        return embeddings

def main():    
    model = BertModel(model_path)
    
    txt_file = "sample_client_texts.txt"
    print(f"Loading texts from {txt_file}...")
    with open(os.path.join(os.path.dirname(__file__), txt_file), 'r') as f:
        texts = f.readlines()
    
    # texts = input("Enter context:")
    texts = texts.split('. ')
    embedded_texts = model.encode(texts)
    
    print(f"Loaded {len(texts)} lines.")

    def print_results(res):
        (closest_texts, closest_similarities) = res
        # Print the closest texts and their similarity scores
        print("Closest texts:")
        for i, text in enumerate(closest_texts):
            print(f"{i+1}. {text} (similarity score: {closest_similarities[i]:.4f})")

    # Define the function to query the k closest texts
    def query(text, k=3, threshold=0.2):
        # Embed the input text
        embedded_text = model.encode(text)
        # Compute the cosine similarity between the input text and all the embedded texts
        similarities = [dot_product(embedded_text, embedded_text_i) / (euclidean_norm(embedded_text) * euclidean_norm(embedded_text_i)) for embedded_text_i in embedded_texts]
        # Sort the similarities in descending order
        sorted_indices = argsort(similarities)[::-1]
        # Return the k closest texts and their similarities
        closest_texts = [texts[i] for i in sorted_indices if similarities[i] > threshold]
        closest_similarities = [similarities[i] for i in sorted_indices[:k] if similarities[i] > threshold]
        return closest_texts, closest_similarities

    test_query = "Should I get health insurance?"
    print(f'Starting with a test query "{test_query}"')
    print_results(query(test_query))

    while True:
        # Prompt the user to enter a text
        input_text = input("Enter a text to find similar texts (enter 'q' to quit): ")
        # If the user enters 'q', exit the loop
        if input_text == 'q':
            break
        # Call the query function to find the closest texts
        print_results(query(input_text))



if __name__ == '__main__':
    main()
