import os
import json

DEFAULT_CONFIG = {
    "batch_size": 32,
    "block_size": 128,
    "max_iters": 3000,
    "eval_interval": 300,
    "eval_iters": 20,
    "learning_rate": 1e-3,
    "min_lr": 1e-4,
    "warmup_iters": 400,
    "weight_decay": 1e-1,
    "n_embd": 256,
    "n_head": 8,
    "n_layer": 6,
    "dropout": 0.1,
    "patience": 5,
    "min_delta": 0.003,
    "karpathy_val_target": 1.47,
    "pos_emb_type": "absolute",
    "top_k": 40,
    "tokenizer": "bpe"
}

def get_latest_checkpoint(runs_dir="runs", default_path="best_model.pt"):
    """
    Finds the latest experiment checkpoint in runs/, Day 2/runs/, Day 1/runs/, or falls back to root default_path.
    Returns (checkpoint_path, model_config_dict).
    """
    search_dirs = [runs_dir, "Day 3/runs", "Day 2/runs", "Day 1/runs"]
    candidate_exps = []
    
    for sdir in search_dirs:
        if os.path.exists(sdir):
            for d in os.listdir(sdir):
                full_p = os.path.join(sdir, d)
                if d.startswith("exp") and os.path.isdir(full_p):
                    candidate_exps.append(full_p)

    candidate_exps.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)

    for exp in candidate_exps:
        ckpt_path = os.path.join(exp, "best_model.pt")
        config_path = os.path.join(exp, "config.json")
        if os.path.exists(ckpt_path):
            cfg = DEFAULT_CONFIG.copy()
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        cfg.update(json.load(f))
                except Exception as e:
                    print(f"⚠️ Warning loading {config_path}: {e}")
            return ckpt_path, cfg

    # Fallback to root model path
    cfg = DEFAULT_CONFIG.copy()
    root_cfg_path = "best_config.json"
    if os.path.exists(root_cfg_path):
        try:
            with open(root_cfg_path, "r") as f:
                cfg.update(json.load(f))
        except Exception:
            pass

    return default_path, cfg

def get_dataset_module(tokenizer_type="char"):
    """ Dynamically imports dataset module based on tokenizer_type ('char' vs 'bpe') """
    if tokenizer_type == "bpe":
        import dataset_bpe as ds
    else:
        import dataset as ds
    return ds

