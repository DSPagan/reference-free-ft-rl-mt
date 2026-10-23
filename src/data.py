"""
Dataset registry shared by main.py, generate_translations.py and decoding_diagnostic.py:
where each dataset's parquet files live per source language, and which split main.py
trains on / intermediate-evals on by default.

Every dataset uses the same layout, {data_dir}/{target_lang}/{split}.parquet with columns
src, ref (written by prepare_data.py for FLORES-200 and prepare_uji.py for UJI).
"""
DATASETS = {
    # FLORES-200: the dev split is both the training set and the sparse intermediate eval set.
    "flores": dict(data_dirs={"en": "data/flores200_eng", "es": "data/flores200_spa"},
                   train_split="dev", eval_split="dev"),
    # UJI va-en news corpus (Valencian references, English source only): a dedicated train
    # split, with the held-out dev split used for the intermediate eval.
    "uji": dict(data_dirs={"en": "data/uji_eng"}, train_split="train", eval_split="dev"),
}


def resolve_data_dir(dataset, source, data_dir=None):
    """The --data_dir override if given, else the dataset's data directory for `source`."""
    if data_dir:
        return data_dir
    dirs = DATASETS[dataset]["data_dirs"]
    if source not in dirs:
        raise ValueError(f"dataset '{dataset}' has no '{source}' source (available: {', '.join(dirs)})")
    return dirs[source]
