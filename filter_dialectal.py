"""
Keeps only the sentences of the UJI Valencian-English corpus whose Valencian side (VA) contains
a word or phrase that is written differently in Catalan, i.e. the "problematic" sentences where
a Catalan-leaning model is most likely to drift away from the Valencian norm.

The Valencian/Catalan pairs come from the AVLizador dictionaries
(third_party/AVLizador/pipeline/02_regles_dialectals/lexic/diccionarios/):
  - lexico_general.json                  general lexicon (nouns, adjectives, numerals, ...)
  - verbos_diferentes_conjugados.json    verbs with a different root (telefonar / trucar)
  - conjugaciones_general.json           verb forms that differ (cante / canti), ~44k rows
By default only the first two are used: the conjugation table pairs subjunctive forms like
'entre', 'sobre' or 'projecte' (Valencian) with 'entri', 'sobri', 'projecti', but those spellings
are ordinary Catalan words in other readings, so it flags a large share of unrelated sentences.
Add `conjugations` to --sources if you want the verb-form markers (sigut, siga, ...) anyway.

A pair is only a marker if the Valencian form is not also a valid Catalan form somewhere in the
dictionaries (e.g. 'mentres' vs 'mentre' is a marker, but a Valencian form that is also another
entry's Catalan form would flag every sentence that uses it), if the two sides differ by more
than case or punctuation, and if the entry is not marked unverified (`_PENDENT`) or homonymous
in the dictionary. Matching is case-insensitive on whole words; multi-word entries
('davall de') must appear as consecutive words. Apostrophes and hyphens split words, so
"l'aigua" and "l’aigua" are the same thing.

Writes the surviving rows to a jsonl with the same fields as the input plus `dialectal_matches`,
the Valencian forms found in the sentence. prepare_uji.py reads that file alongside the full one.

Usage:
    uv run python filter_dialectal.py
    uv run python filter_dialectal.py --sources lexic verbs conjugations --min_matches 2
"""
import argparse
import json
import os
import re

DICT_DIR = "third_party/AVLizador/pipeline/02_regles_dialectals/lexic/diccionarios"
SOURCES = {
    "lexic": "lexico_general.json",
    "verbs": "verbos_diferentes_conjugados.json",
    "conjugations": "conjugaciones_general.json",
}
WORD = re.compile(r"[^\W\d_]+")
HOMONYMS = {"homonimia_lexica", "colision_homonima"}


def tokenize(text):
    return tuple(WORD.findall(text.lower()))


def usable(entry, source):
    """False for entries the dictionaries themselves flag as unverified or ambiguous."""
    if entry.get("problematica") or "PENDENT" in str(entry.get("tipo")) or entry.get("problema") in HOMONYMS:
        return False
    return not (source == "conjugations" and entry.get("problema"))


def load_markers(dict_dir, sources):
    """Returns {valencian token tuple: surface form} for the Valencian forms that differ from
    Catalan and are not themselves Catalan forms."""
    pairs = []
    for name in sources:
        with open(os.path.join(dict_dir, SOURCES[name]), encoding="utf-8") as f:
            for e in json.load(f):
                va, ca = e.get("valenciano"), e.get("catalan")
                if isinstance(va, str) and isinstance(ca, str) and usable(e, name):
                    pairs.append((va, tokenize(va), tokenize(ca)))
    catalan_forms = {ca for _, _, ca in pairs}
    markers = {}
    for va, va_tok, ca_tok in pairs:
        if va_tok and va_tok != ca_tok and va_tok not in catalan_forms:
            markers.setdefault(va_tok, va)
    return markers


def find_matches(text, markers, max_len):
    """Valencian marker forms in `text`, in order of appearance, without repeats."""
    tokens = tokenize(text)
    found = {}
    for i in range(len(tokens)):
        for n in range(1, min(max_len, len(tokens) - i) + 1):
            form = markers.get(tokens[i:i + n])
            if form is not None:
                found.setdefault(form, None)
    return list(found)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/va_en_uji/va-en-sentence.length.ner.curated.jsonl")
    p.add_argument("--output", default="data/va_en_uji/va-en-sentence.length.ner.curated.dialectal.jsonl")
    p.add_argument("--dict_dir", default=DICT_DIR)
    p.add_argument("--sources", nargs="+", choices=list(SOURCES), default=["lexic", "verbs"],
                   help="Which AVLizador dictionaries provide the Valencian/Catalan pairs.")
    p.add_argument("--min_matches", type=int, default=1,
                   help="Keep sentences with at least this many distinct marker forms.")
    args = p.parse_args()

    markers = load_markers(args.dict_dir, args.sources)
    max_len = max(len(t) for t in markers)
    print(f"{len(markers)} Valencian marker forms from {', '.join(args.sources)} (longest: {max_len} words)")

    n_total = n_kept = 0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.input, encoding="utf-8-sig") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            n_total += 1
            va = r.get("VA")
            matches = find_matches(va, markers, max_len) if isinstance(va, str) else []
            if len(matches) >= args.min_matches:
                fout.write(json.dumps({**r, "dialectal_matches": matches}, ensure_ascii=False) + "\n")
                n_kept += 1

    print(f"Kept {n_kept} of {n_total} sentences ({n_kept / max(n_total, 1):.1%}) -> {args.output}")


if __name__ == "__main__":
    main()
