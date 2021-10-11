import argparse
import warnings

from src.trainer import *

warnings.simplefilter("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_str', type=str, default='cora')
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate_decay_patience', type=int, default=10)
    parser.add_argument('--learning_rate_decay_factor', type=float, default=0.8)
    parser.add_argument('--model', type=str, default='HeteroGAT')
    parser.add_argument('--dataset', type=str, default='data/word_data/hep-small/hep-small.pickle.bin')
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--out_dim', type=int, default=7)
    parser.add_argument('--aggr_func', type=str, default='mean',
                        choices=["mean", "sum", "linear", "mlp", "attention"])
    parser.add_argument('--node_feature', type=str, default='one_hot', choices=['one_hot', 'message_passing','bert','raw'])
    parser.add_argument('--word_feature', type=str, default='w2v', choices=['w2v', 'jose'])
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--num_head', type=int, default=2)
    parser.add_argument('--residual', type=int, default=0)
    args = parser.parse_args()
    args.verbose = bool(args.verbose)
    args.residual = bool(args.residual)
    print(args)
    if args.model == "HeteroGCN":
        hetero_gcn(args)
    elif args.model == "HeteroGAT":
        hetero_gat(args)
    elif args.model == "VanillaGCN":
        vanilla_gcn(args)
    elif args.model == "VanillaGAT":
        vanilla_gat(args)
