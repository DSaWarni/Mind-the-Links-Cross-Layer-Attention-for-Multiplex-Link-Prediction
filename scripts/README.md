
# Scripts

Examples (adjust paths to your dataset):

```bash
python -m lpmp.cli sle       --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges       --node2vec ./data_files/CS-Aarhus_multiplex/Node2Vec_layer_embedding.pkl       --target-layer "friendship"       --batch-size 512 --epochs 50 --lr 1e-3

python -m lpmp.cli gat       --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges       --node2vec ./data_files/CS-Aarhus_multiplex/Node2Vec_layer_embedding.pkl       --target-layer "friendship"       --batch-size 512 --epochs 50 --lr 1e-3       --gat-heads 4 --gat-dropout 0.1 --tfm-layers 2 --tfm-heads 4 --tfm-ff 512 --tfm-dropout 0.1
```


---

## Compute embeddings on the fly (no precomputed .pkl)

**Node2Vec (per layer):**
```bash
python -m lpmp.cli sle       --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges       --embed-backend node2vec --emb-dim 128       --n2v-p 1.0 --n2v-q 1.0 --n2v-walk-len 50 --n2v-walks-per-node 10       --sgns-window 10 --sgns-epochs 1 --sgns-batch 8192 --sgns-lr 1e-3 --sgns-neg-k 1       --target-layer 1 --epochs 30
```

**Corewalk (core2vec-inspired bias):**
```bash
python -m lpmp.cli gat       --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges       --embed-backend corewalk --emb-dim 128       --core-temp 1.0 --core-walk-len 40 --core-walks-per-node 10       --sgns-window 10 --sgns-epochs 1 --sgns-batch 8192 --sgns-lr 1e-3 --sgns-neg-k 1       --target-layer 2 --epochs 30
```

**Random embeddings (sanity baseline):**
```bash
python -m lpmp.cli sle       --edges ./data_files/CS-Aarhus_multiplex/CS-Aarhus_multiplex.edges       --embed-backend random --emb-dim 128       --target-layer 3 --epochs 30
```

**Cache computed embeddings to a .pkl** for reuse:
```bash
--save-embeddings ./data_files/CS-Aarhus_multiplex/Node2Vec_layer_embedding.pkl
```
