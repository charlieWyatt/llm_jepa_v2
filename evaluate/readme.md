# Evaluate:
## Evaluate Longformer:
```
python experiments/llm_jepa_v2/evaluate/imdb_longformer_evaluate.py \
    --model-type longformer \
    --checkpoint /g/data/oy87/cw9909/llm_jepa/checkpoints/longformer/checkpoint-20000
```

## Evaluate JEPA:
```
python experiments/llm_jepa_v2/evaluate/imdb_longformer_evaluate.py     --model-type jepa     --checkpoint /g/data/oy87/cw9909/llm_jepa/checkpoints/
llm_jepa/checkpoint_step_100000.pt
```