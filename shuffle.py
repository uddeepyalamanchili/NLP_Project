import torch
import os
from concurrent.futures import ThreadPoolExecutor
import time

# Function to jumble a single sentence using PyTorch and GPU
def jumble_sentence_with_torch(sentence, device):
    words = sentence.split()  # Split the sentence into words
    words_tensor = torch.tensor([i for i in range(len(words))], device=device)  # Create tensor of word indices
    
    # Shuffle the word indices on GPU
    shuffled_indices = words_tensor[torch.randperm(words_tensor.size(0))]

    # Reorder words based on shuffled indices
    jumbled_words = [words[i.item()] for i in shuffled_indices]
    return ' '.join(jumbled_words)

# Function to jumble the entire corpus using PyTorch with GPU in batches
def jumble_corpus_with_torch(corpus, device, batch_size=5000):
    jumbled_corpus = [None] * len(corpus)  # Initialize a list to hold the jumbled sentences in correct order
    
    # Process the corpus in batches
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i + batch_size]
        jumbled_batch = [jumble_sentence_with_torch(sentence, device) for sentence in batch]
        
        # Store the jumbled sentences in the original order
        for j, sentence in enumerate(jumbled_batch):
            jumbled_corpus[i + j] = sentence
        
        # Optionally, write batch results to a file instead of storing in memory
        save_batch_to_file(jumbled_batch)
    
    return jumbled_corpus

# Function to read sentences from a file and return them as a list
def read_sentences_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()  # Read the entire file content
        sentences = content.split(' .')  # Split sentences by ' .'
        return sentences

# Function to save the jumbled batch to a file to avoid memory overload
def save_batch_to_file(batch, output_file='output_shuffle.txt'):
    with open(output_file, 'a', encoding='utf-8') as file:
        for sentence in batch:
            file.write(sentence + ' .\n')

# Function to read and process sentences in parallel
def process_sentences_parallel(file_path, device, batch_size=5000):
    sentences = read_sentences_from_file(file_path)
    total_sentences = len(sentences)

    # Process in parallel using ThreadPoolExecutor to improve I/O speed
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = []
        start_time = time.time()

        # Split into smaller chunks and process each chunk in parallel
        for i in range(0, total_sentences, batch_size):
            batch = sentences[i:i + batch_size]
            results.append(executor.submit(jumble_corpus_with_torch, batch, device, batch_size))
        
        # Wait for all tasks to complete and gather results
        jumbled_sentences = []
        for future in results:
            jumbled_sentences.extend(future.result())
        
        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds")
        
        return jumbled_sentences

def write_to_file(filename, sentences):
    with open(filename, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            file.write(sentence + ' .\n')  # Writing each sentence followed by a period and new line

# Main function to process the dataset
def main():
    # file_path = "Testing/untitled.src"  # Path to your file
    file_path = "../NLP/data/train_merge.src"  # Path to your file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process the sentences in parallel and jumble them in batches
    jumbled_sentences = process_sentences_parallel(file_path, device, batch_size=5000)

    # Output the first few jumbled sentences (for demonstration)
    for sentence in jumbled_sentences[:10]:  # Adjust the number as needed
        print(sentence)
    # Function to write the list contents to a file


    # Call the function to write jumbled_sentences to a file
    file_path = 'jumbled_sentences.txt'  # Output file path
    write_to_file(file_path, jumbled_sentences)

if __name__ == "__main__":
    main()
