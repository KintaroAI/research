"""Synthetic facts (task 00007): invented people, each with one attribute token, as sentences for a GPT-2 tokenizer.

A fact is "<|endoftext|>{name} was born in{city}." The city is a single GPT-2 token of the form " Xxxxx" that never occurs
in TinyStories, so a model trained on TinyStories has no prior for it. 256 cities serve the ordinary facts (chance 1 in
256); 32 more are kept for the reverse facts, one person each.
"""
import re

import numpy as np
import tiktoken

ENC = tiktoken.get_encoding("gpt2")
EOT = ENC.eot_token
SYL = ("zor van kel mir eth qua lis dor fen gal hux ith jor kam lun mos nev ory pax rin sul tam ugo vex wyn yal zed bro "
       "cra dri fla gro pli sko tri vor xan").split()


def ts_tokens(path):
    """A TinyStories .bin of llm.c (a 1 KB header, then uint16 GPT-2 tokens) as a memmap."""
    return np.memmap(path, dtype=np.uint16, mode="r", offset=1024)


def make(seed, n_people, val_path):
    """The cities [288] (token ids) and n_people names, none of which occurs in TinyStories' validation text."""
    rng = np.random.default_rng(seed)
    cnt = np.bincount(np.asarray(ts_tokens(val_path)), minlength=ENC.n_vocab)
    ok = [t for t in range(ENC.n_vocab) if cnt[t] == 0 and re.fullmatch(r" [A-Z][a-z]{4,8}", ENC.decode([t]))]
    cities = rng.choice(ok, 288, replace=False)
    names, seen = [], set()
    while len(names) < n_people:
        first = "".join(rng.choice(SYL, 2)).capitalize()
        last = "".join(rng.choice(SYL, 2)).capitalize()
        n = first + " " + last
        if n not in seen:
            seen.add(n)
            names.append(n)
    return cities, names


class Facts:
    """Token sequences of the facts and of their probes. kind: "plain" (city among cities[:256]) or "reverse" (its own city
    among cities[256:])."""

    def __init__(self, names, cities, city_of):
        self.names, self.cities, self.city_of = names, np.asarray(cities), np.asarray(city_of)
        self.sentence = [[EOT] + ENC.encode(f"{n} was born in") + [int(self.cities[c])] + ENC.encode(".")
                         for n, c in zip(names, city_of)]
        self.exact = [[EOT] + ENC.encode(f"{n} was born in") for n in names]
        self.para = [[EOT] + ENC.encode(f"Where was {n} born? In") for n in names]
        self.name_tokens = [ENC.encode(" " + n) for n in names]

    def reverse_prompt(self, i):
        return [EOT] + ENC.encode("Who was born in") + [int(self.cities[self.city_of[i]])] + ENC.encode("? It was")


def check_absent(names, text):
    """The names that occur in a text (should be none)."""
    return [n for n in names if n in text]
