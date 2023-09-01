import json
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize

def plot_histgram(input_file, output_file):
    # Load the data from the input JSON file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Initialize an empty list to store the number of tokens for each description
    token_counts = []

    for item in data:
        # Tokenize the description and get the number of tokens
        token_count = len(word_tokenize(item['description']))
        # Append the number of tokens to the list
        token_counts.append(token_count)

    # Plot the histogram
    plt.hist(token_counts, bins=50, edgecolor='black')
    plt.title("Number of tokens in 'description'")
    plt.xlabel('Number of tokens')
    plt.ylabel('Frequency')
    plt.savefig(output_file)

if __name__ == '__main__':
    input_file = 'scanrefer/ScanRefer_downsampled_val.json'
    output_file = 'scanrefer/val_ds_word_count.png'
    plot_histgram(input_file, output_file)