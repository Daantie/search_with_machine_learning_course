import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])

# levels from categories
all_done = False
levels = [-1] * len(parents)
categories_copy = categories
parents_copy = parents
curr_level = 0
max_level = -1
while not all_done:
    cur_leaf_categories = [value for value in categories_copy if value not in parents_copy]
    leaf_indices = []
    for leaf_cat in cur_leaf_categories:
        actual_idx = categories.index(leaf_cat)
        copy_leaf_idx = categories_copy.index(leaf_cat)
        leaf_indices.append(copy_leaf_idx)
        levels[actual_idx] = curr_level
    categories_copy = [j for i, j in enumerate(categories_copy) if i not in leaf_indices]
    parents_copy = [j for i, j in enumerate(parents_copy) if i not in leaf_indices]
    if len(categories_copy) == 0:
        all_done = True
    else:
        curr_level += 1

max_level = curr_level

parents_df = pd.DataFrame(list(zip(categories, parents, levels)), columns=['category', 'parent', 'level'])
print(parents_df)

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
queries_df['query'] = queries_df['query'].str.lower()
queries_df['tokens'] = queries_df['query'].str.split()
queries_df['stemmed_tokens'] = queries_df['tokens'].apply(lambda x: [stemmer.stem(y) for y in x])
queries_df['query'] = queries_df['stemmed_tokens'].str.join(' ')

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
queries_df_with_counts = queries_df.groupby('category').size().reset_index(name='count')
queries_df_merged = queries_df.merge(queries_df_with_counts, how='left', on='category').merge(parents_df, how='left',
                                                                                              on='category')
print("Number of sub-threshold categories: " + str(len(queries_df_merged[queries_df_merged['count'] < min_queries])))
curr_level = 0
while len(queries_df_merged[queries_df_merged['count'] < min_queries]) > 0:
    print(f"Curr level: {curr_level}")
    queries_df_merged.loc[
        (queries_df_merged['count'] < min_queries) & (queries_df_merged['level'] == curr_level), 'category'] = \
    queries_df_merged['parent']
    # queries_df_merged.loc[queries_df_merged['count'] < min_queries, 'category'] = queries_df_merged['parent']
    queries_df = queries_df_merged[['category', 'query']]
    queries_df = queries_df[queries_df['category'].isin(categories)]
    queries_df_with_counts = queries_df.groupby('category').size().reset_index(name='count')
    queries_df_merged = queries_df.merge(queries_df_with_counts, how='left', on='category').merge(parents_df,
                                                                                                  how='left',
                                                                                                  on='category')
    print(f"Number of unique categories: {queries_df_merged['category'].nunique()}")
    print(
        "Number of sub-threshold categories: " + str(len(queries_df_merged[queries_df_merged['count'] < min_queries])))
    curr_level += 1

print(f"Number of unique categories: {queries_df['category'].nunique()}")
# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE,
                              index=False)
