import json
import random
from pathlib import Path


def load_pairs_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def select_pair_items(doc, limit=None, sample_mode="head", sample_seed=0):
    pairs = list(doc.get("pairs", []))
    if limit is None or int(limit) <= 0 or int(limit) >= len(pairs):
        return pairs

    limit = int(limit)
    if sample_mode == "random":
        rng = random.Random(sample_seed)
        indexed = list(enumerate(pairs))
        chosen = rng.sample(indexed, limit)
        chosen.sort(key=lambda item: item[0])
        return [item for _, item in chosen]

    return pairs[:limit]

