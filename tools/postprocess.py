import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='post',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='hep-small')
    parser.add_argument('--in_file', default='phrase_text.txt')
    parser.add_argument('--out_file', default='all_text.txt')
    args = parser.parse_args()

    dataset = Path(args.dataset)
    with open(dataset / args.in_file, 'r') as f:
        phrased_data = [line.strip() for line in f]

    unphrased_data = []
    for line in phrased_data:
        unphrased_data.append(line.replace("_", " "))
    with open(dataset / args.out_file, "w") as f:
        f.write("\n".join(phrased_data + unphrased_data))
    words = []
    word_links = []

    with open(dataset / "AutoPhrase_tagged.txt", "r") as f:  # only consider the phrases that appear in the corpus
        lines = f.readlines()
        tagged_phrases = {line.strip() for line in lines}
    with open(dataset / "AutoPhrase.txt", "r") as f:  # only consider the phrases that appear in the corpus
        lines = f.readlines()
        phrase2score={}
        for line in lines:
            score, phrase = line.strip().split("\t")
            phrase2score[phrase]=float(score)
    phrase2score = list(phrase2score.keys())
    for phrase in phrase2score:
        phrase = phrase.split(" ")
        phrase_raw = "_".join(phrase)
        if len(phrase) == 1:
            for w in phrase:
                if w not in words:
                    words.append(w)
        else:
            phrase.append(phrase_raw)
            for w in phrase:
                if w not in words:
                    words.append(w)
            for w1, w2 in combinations(phrase, 2):
                word_links.append(f"{w1}\t{w2}")
    with open(dataset / "word_nodes.txt", "w") as f:
        f.write("\n".join(words))
    with open(dataset / "word_links.txt", "w") as f:
        f.write("\n".join(word_links))
    word2id = {w: i for i, w in enumerate(words)}
    with open(dataset / "nodes.txt", "r") as f:
        nodes = f.readlines()
        nodes = [n.strip() for n in nodes]
    embeddings = np.zeros((len(nodes), len(word2id)), dtype=np.int)
    for node, line in enumerate(phrased_data):
        tokens = line.split(" ")
        for tok in tokens:
            if tok in word2id:
                embeddings[node][word2id[tok]] = 1
    # with open(dataset/"features.emb","wb") as f:
    #     pkl.dump(embeddings,f)
    with open(dataset / "features.txt", "w") as f:
        for node, feature in zip(nodes, embeddings):
            emb = " ".join(map(str, feature.tolist()))
            f.write(f"{node}\t{emb}\n")
