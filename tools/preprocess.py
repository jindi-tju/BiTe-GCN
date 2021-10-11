import csv
import numpy as np
import re
from os.path import join
from tqdm import tqdm


def read_file(data_dir, in_file='text.txt'):
    data = []
    filename = join(data_dir, in_file)
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            data.append(line.strip())
    return data


def process_period(line):
    tokens = line.split(' ')
    for i, token in enumerate(tokens):
        separate = token.split('.')
        num_dots = len(separate) - 1
        if num_dots == 0:
            continue
        new_token = ''
        for j, letter in enumerate(separate):
            if j == len(separate) - 1:
                new_token += letter
            else:
                if len(letter) == 0 and len(separate[j+1]) > 0:
                    new_token += letter + '. '
                elif len(separate[j+1]) == 0 and len(letter) > 0:
                    new_token += letter + ' .'
                elif len(letter) > 1 or len(separate[j+1]) > 1:
                    new_token += letter + ' . '
                else:
                    new_token += letter + '.'
        tokens[i] = new_token
    ret = ' '.join(tokens)
    return ret


def clean_str(string):
    string = process_period(string)
    string = re.sub(r"<.*?/>", " ", string)
    string = re.sub(r"\\n", " ", string)
    string = re.sub(r"\\t", " ", string)
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'\'", "\"", string)
    string = re.sub(r"\"", " \" ", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\$", " $ ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def preprocess_doc(data):
    for i in tqdm(range(len(data))):
        s = data[i].strip()
        data[i] = clean_str(s)
    return data


def clean_file(train_data, out_dir='./', out_file='./cleaned.txt'):
    data = preprocess_doc(train_data)
    with open(join(out_dir, out_file), 'wt') as outfile:
        for line in data:
            outfile.write(line + '\n')
    return


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='pre',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='yelp')
    parser.add_argument('--in_file', default='text.txt')
    parser.add_argument('--out_file', default='cleaned.txt')
    args = parser.parse_args()
    dataset = args.dataset

    data = read_file(dataset, in_file=args.in_file)
    clean_file(data, out_file=args.out_file)
