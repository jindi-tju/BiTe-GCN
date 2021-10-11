import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from numpy.linalg import norm
from tqdm import tqdm

stopwords = set(stopwords.words("english"))
MAX_WORD_NUMBER = 10000
JOSE_HIGH = W2V_HIGH = 0.95
JOSE_LOW = W2V_LOW = 0.4
BERT_LOW = 0.4
BERT_HIGH = 0.95


def cosine(x, y):
    return np.dot(x, y) / (norm(x) * norm(y))


def construct_word_net(args):
    dataset = Path(args.dataset)
    with open(dataset / "nodes.txt") as f:
        nodes = [line.strip() for line in f]
    with open(dataset / "jose.vec") as f:
        jose_embeddings = KeyedVectors.load_word2vec_format(f, binary=False)
    with open(dataset / "all.vec") as f:
        w2v_embeddings = KeyedVectors.load_word2vec_format(f, binary=False)
    phrase2score = {}
    with open(dataset / "AutoPhrase" / "AutoPhrase.txt") as f:
        for line in f:
            score, phrase = line.strip().split("\t")
            phrase = phrase.replace(" ", "_")
            if phrase in jose_embeddings and phrase in w2v_embeddings:
                phrase2score[phrase] = float(score)
    print("data loaded")
    phrase2score_list = sorted(phrase2score.items(), key=lambda x: x[1], reverse=True)
    if len(phrase2score_list) > MAX_WORD_NUMBER:
        phrase2score_list = phrase2score_list[:MAX_WORD_NUMBER]
    phrase_list = [p for p, s in phrase2score_list]
    word_cnt = {}
    word2phrase = {}
    for p in phrase_list:
        words = p.split("_")
        for w in set(words):
            if w in word_cnt:
                word_cnt[w] += 1
                word2phrase[w].append(p)
            else:
                word_cnt[w] = 1
                word2phrase[w] = [p]
    word_links = set()
    for w, cnt in word_cnt.items():
        if cnt > 1 and w not in stopwords:
            for p1, p2 in combinations(word2phrase[w], 2):
                word_links.add(f"{p1}\t{p2}")
    print("write output files")
    with open(dataset / "word_nodes.txt", "w") as f:
        f.write("\n".join(phrase_list))
    with open(dataset / "word_links.txt", "w") as f:
        f.write("\n".join(word_links))
    word2id = {w: i for i, w in enumerate(phrase_list)}
    with open(dataset / "nodes.txt", "r") as f:
        nodes = f.readlines()
        nodes = [n.strip() for n in nodes]
    embeddings = np.zeros((len(nodes), len(word2id)), dtype=np.int)
    print("generate one hot features")
    with open(dataset / "phrase_text.txt") as f:
        phrased_data = f.readlines()
    for node, line in enumerate(phrased_data):
        tokens = line.strip().split(" ")
        for tok in tokens:
            if tok in word2id:
                embeddings[node][word2id[tok]] = 1
    # with open(dataset/"features.emb","wb") as f:
    #     pkl.dump(embeddings,f)
    with open(dataset / "features.txt", "w") as f:
        for node, feature in zip(nodes, embeddings):
            emb = " ".join(map(str, feature.tolist()))
            f.write(f"{node}\t{emb}\n")


def refine_word_net(args):
    dataset = Path(args.dataset)
    with open(dataset / "jose.vec") as f:
        jose_embeddings = KeyedVectors.load_word2vec_format(f, binary=False)
    with open(dataset / "all.vec") as f:
        w2v_embeddings = KeyedVectors.load_word2vec_format(f, binary=False)
    jose_add = set()
    jose_del = set()
    w2v_add = set()
    w2v_del = set()
    jose_word_links = []
    w2v_word_links = []
    with open(dataset / "word_nodes.txt") as f:
        word_list = [w.strip() for w in f.readlines()]
    word_links = []
    with open(dataset / "word_links.txt") as f:
        for line in f:
            w1, w2 = line.strip().split("\t")
            word_links.append((w1, w2))

    for w1, w2 in word_links:
        if jose_embeddings.similarity(w1, w2) > JOSE_LOW:
            jose_word_links.append((w1, w2))
        else:
            jose_del.add((w1, w2, float(jose_embeddings.similarity(w1, w2))))
        if w2v_embeddings.similarity(w1, w2) > W2V_LOW:
            w2v_word_links.append((w1, w2))
        else:
            w2v_del.add((w1, w2, float(w2v_embeddings.similarity(w1, w2))))
    print("refine")
    for w in tqdm(word_list):
        for w2, score in jose_embeddings.similar_by_word(word=w, topn=1000):
            # print(f"jose: {w2},{score}")
            if score > JOSE_HIGH and w2 in word_list and (w, w2) not in jose_word_links:
                jose_word_links.append((w, w2))
                jose_add.add((w, w2, float(score)))
            else:
                break
        for w2, score in w2v_embeddings.similar_by_word(word=w, topn=1000):
            # print(f"w2v: {w2},{score}")
            if score > W2V_HIGH and w2 in word_list and (w, w2) not in w2v_word_links:
                w2v_word_links.append((w, w2))
                w2v_add.add((w, w2, float(score)))
            else:
                break
    with open(dataset / "jose_word_links.txt", "w") as f:
        for w1, w2 in jose_word_links:
            f.write(f"{w1}\t{w2}\n")
    with open(dataset / "w2v_word_links.txt", "w") as f:
        for w1, w2 in w2v_word_links:
            f.write(f"{w1}\t{w2}\n")
    with open(dataset / "check_word_refine.json", "w") as f:
        json.dump({"jose_add": list(jose_add),
                   "jose_del": list(jose_del),
                   "w2v_add": list(w2v_add),
                   "w2v_del": list(w2v_del),
                   }, f, indent=2)


def refind_doc_net(args):
    dataset = Path(args.dataset)
    with open(dataset / "nodes.txt") as f:
        nodes = [line.strip() for line in f]
    node2id = {n: nid for nid, n in enumerate(nodes)}
    with open(dataset / "edges.txt") as f:
        edges = []
        for line in f:
            s, t = line.strip().split()
            edges.append((s, t))
    if args.dataset == "cora_enrich":
        with open(dataset / "bert_embs.txt") as f:
            bert_embs = np.loadtxt(f)
    else:
        with open(dataset / "bert_embs.txt") as f:
            node_cnt, emb_dim = f.readline().strip().split(", ")
            f.readline()
            node2bertemb = {}
            for line in f:
                node, _, emb = line.strip().split("\t")
                node2bertemb[node] = np.array(emb.split(' '), dtype=float)
        bert_embs = np.array([node2bertemb[node] for node in nodes])
    bert_links = set()
    emb_norms = norm(bert_embs, axis=1)
    norm_matrix = np.expand_dims(emb_norms, 1) @ np.expand_dims(emb_norms, 0)
    cos_matrix = (bert_embs @ (bert_embs.T)) / norm_matrix - 5 * np.eye(emb_norms.shape[0])
    indices = np.where(cos_matrix > BERT_HIGH)
    bert_refine_add = []
    bert_refine_del = []
    for x, y in tqdm(zip(indices[0], indices[1])):
        bert_links.add((nodes[x], nodes[y]))
        bert_refine_add.append((nodes[x], nodes[y], cos_matrix[x][y]))

    for d1, d2 in tqdm(edges):
        if d1 in nodes and d2 in nodes:
            tmp_score = cosine(bert_embs[node2id[d1]], bert_embs[node2id[d2]])
        else:
            continue
        if tmp_score > BERT_LOW:
            bert_links.add((d1, d2))
        else:
            bert_refine_del.append((d1, d2, tmp_score))
    with open(dataset / "check_bert_refine.json", "w") as f:
        json.dump({"add": bert_refine_add, "del": bert_refine_del}, f, indent=2)

    with open(dataset / "bert_edges.txt", "w") as f:
        for d1, d2 in bert_links:
            f.write(f"{d1}\t{d2}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='cora_enrich')
    args = parser.parse_args()
    print(args.dataset)
    # construct_word_net(args)
    # refind_doc_net(args)
    refine_word_net(args)
