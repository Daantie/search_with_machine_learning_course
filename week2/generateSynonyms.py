import argparse
import fasttext
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='Generate synonyms for the provided words')
general = parser.add_argument_group("general")
general.add_argument("--modelpath", default="/workspace/datasets/fasttext/title_model.bin", help="The model to be used to generate synonyms")
general.add_argument("--wordlistpath", default="/workspace/datasets/fasttext/top_words.txt", help="The list of words for which to generate synonyms")
general.add_argument("--similaritythreshold", default=0.75, help="The similarity threshold below which a word will NOT be considered a synonym")
general.add_argument("--output", default="/workspace/datasets/fasttext/synonyms.csv", help="The file to output the synonyms")

args = parser.parse_args()

# Ensure the output directory exists
path = Path(args.output)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
    os.mkdir(output_dir)

# The fasttext model we'll use to generate synonyms
model = fasttext.load_model(args.modelpath)

wordlist = open(args.wordlistpath)

# For every word in the input list, generate a list of synonyms using the provided model.
print(f"Writing output to {args.output}")
with open(args.output, 'w') as output:
    while True:

        word = wordlist.readline()

        if not word:
            break

        # Get this word's nearest neighbours
        nearest_neighbours = model.get_nearest_neighbors(word)

        synonyms = []

        # To be considered a synonym, the word's similarity rating must be greater than the provided threshold
        for neighbour in nearest_neighbours:
            similarity_rating = neighbour[0]
            synonym = neighbour[1]

            if similarity_rating >= args.similaritythreshold:
                synonyms.append(synonym)

        # Only output words with synonyms
        if len(synonyms) > 1:
            output.write(f"{','.join(synonyms)}\n")
