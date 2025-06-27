import re

def build_vocab(lines, min_freq=1):
    vocab = {"<pad>": 0, "<unk>": 1, "<sep>": 2, "<eos>": 3}
    freq = {}
    for line in lines:
        tokens = re.findall(r"\w+|\S", line.lower())
        for tok in tokens:
            freq[tok] = freq.get(tok, 0) + 1
    for tok, f in freq.items():
        if f >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab

def tokenize(text, vocab):
    tokens = re.findall(r"\w+|\S", text.lower())
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
