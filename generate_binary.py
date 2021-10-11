import argparse
from tqdm import tqdm
from src.dataset import GraphDataset

if __name__ == '__main__':
    print("Generating pickle.bin file")
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default='hep-small')
    args = parser.parse_args()
    GraphDataset(dataset_str=args.dataset)
