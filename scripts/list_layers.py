
import argparse
from lpmp.data import load_edges

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edges", required=True)
    p.add_argument("--layer-col", default=None)
    p.add_argument("--start-col", default=None)
    p.add_argument("--end-col", default=None)
    p.add_argument("--id-base", type=int, default=1, choices=[0,1])
    p.add_argument("--has-header", action="store_true")
    args = p.parse_args()
    d = load_edges(args.edges, layer_col=args.layer_col, start_col=args.start_col, end_col=args.end_col, id_base=args.id_base, has_header=args.has_header)
    print("Layers:", d.layers)
    print("Num nodes:", d.num_nodes)
    for L in d.layers:
        print(f"{L}: |E|={len(d.edge_sets[L])}")
if __name__ == "__main__":
    main()
