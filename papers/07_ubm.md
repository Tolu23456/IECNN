# 07. UBM: Universal Basemapping

UBM is IECNN's tokenizer. Every piece of text is built from numbered **bases**: raw bytes at the bottom, merged pieces above them, and promoted frequent words at the top. A new word is never "unknown": it always decomposes into bases that already exist.

Status: draft v0.1. Decision D10: UBM's main purpose is handling new words; the layers and regions drive vocabulary promotion at checkpoints.

---

## 1. Honest positioning

As described so far (letters get numbers; shared pieces like `cat` inside `catography` reuse the same ID; frequent words get promoted to bases), UBM is essentially **byte-level byte-pair encoding (BPE)**. In GPT-2-style byte-level BPE the 256 byte values are IDs 0 to 255 and the first learned merge gets ID 256.

That is not a flaw. It is a solved, reliable base. UBM's own contribution has to come from:

1. the **promotion rule** (driven by dot agreement in the spelling-level region, not frequency alone),
2. **composition-based representations** for new bases,
3. a **checkpointed, reversible** vocabulary growth process,
4. a fairness check for languages that general tokenizers handle poorly.

Related work to be aware of (verify before citing): dynamic chunking models such as H-Net and Byte Latent Transformer, and vocabulary expansion methods that initialize a new token's embedding from its constituent tokens.

## 2. Vocabulary layout

| ID range | Contents |
|---|---|
| 0 to 255 | Raw bytes (the 256 possible byte values) |
| 256 to 263 | Specials: `<pad>`, `<bos>`, `<eos>`, `<sep>`, `<mask>`, and three reserved |
| 264 onward | Learned bases (merges and promotions), in order of creation |

- IDs are **never reused and never removed**. A rolled-back base becomes a tombstone (kept in the table, flagged inactive).
- The example from the original idea ("cat = 257") would correspond to the first learned merge at ID 264 in this layout, because of the specials block.

## 3. Base record

Every base has:

| Field | Meaning |
|---|---|
| `id` | Integer ID |
| `bytes` | The byte string the base stands for |
| `parts` | `[left_id, right_id]` for merged bases; empty for raw bytes |
| `birth_step` | Training step when created (0 for bytes) |
| `origin` | `merge` or `promotion` |
| `count` | Occurrence count when created |
| `active` | False if tombstoned |

Because every non-byte base records its two parts, any base can be expanded into bytes by walking the tree. That is the "never unknown" guarantee.

## 4. Encoding

1. **Normalize** text to Unicode NFC. (Some languages, including Yoruba with dot-below plus tone marks, have combinations with no single composed form. NFC leaves those as combining sequences; bytes handle them. Test this explicitly.)
2. **Pre-tokenize** into chunks using a regular expression: words (with an attached leading space marker), numbers, punctuation runs, whitespace runs. Merges never cross chunk boundaries.
3. Convert each chunk to its UTF-8 bytes.
4. **Apply merges** greedily by creation order (lowest ID first), as in standard BPE, until none apply.
5. Output the list of IDs.

Decoding concatenates the bytes of each ID and decodes UTF-8. Invalid byte sequences produced by generation are decoded with replacement characters.

**Invariant**: `decode(encode(text)) == normalize(text)` for all valid input.

## 5. Handling new words

A word the vocabulary has never seen is encoded as the longest known bases that fit, down to single bytes if needed. Example: `catography` becomes the base for `cat` plus bases for `ography` (or its pieces). No unknown token is ever needed.

The model's input representation of the word is the composition of its parts' embeddings automatically (each part is a token). When a new word keeps appearing, it is promoted to its own base (section 6) and its embedding is initialized from its parts.

## 6. Vocabulary growth

### 6.1 v0: frequency promotion (build this first)

Standard BPE training: repeatedly merge the most frequent adjacent pair. Fixed vocabulary size, trained once before model training. Use an existing implementation for the baseline.

### 6.2 v1: checkpoint promotion driven by the model (D10)

Vocabulary changes happen only at **checkpoints**, in small batches, with a settling phase and a rollback test.

1. **Train normally** for a fixed stretch (default 5k steps) while logging **candidates**. A candidate is a short sequence of existing bases (length 2 to 6) that the spelling-level region (region 1) predicted with high agreement, many times. The log stores: the sequence, how many times seen, mean plurality agreement, mean consensus confidence.
2. **Rank** candidates by a score combining frequency, agreement, and confidence. Keep only the top K (default K = 16). A small K protects stability.
3. **Create** a new base ID for each survivor, with `parts` set to the two-part split that reproduces the sequence, `origin = promotion`.
4. **Initialize** its embedding as the **mean of its constituents' embeddings** (so behaviour barely changes at the moment of promotion). The output side shares the tied embedding. A fallback is to initialize from the first constituent for the output only; compare in a small test.
5. **Retokenize** the training data so occurrences of the sequence now use the new base.
6. **Settling phase**: for S steps (default 500), update only the new bases' embedding rows (everything else frozen).
7. **Rollback test**: compare held-out **bits-per-character** (section 7) before and after. If a promotion made it worse by more than epsilon, tombstone that base and restore the old tokenization for it.
8. **Resume** normal training; repeat at the next checkpoint.

Cheaper early version: promote by frequency plus agreement and run the full rollback test only periodically.

### 6.3 Why batches and phases

Changing the vocabulary while dots are learning can scramble what they have learned. Small batches, mean-of-parts initialization, a frozen settling phase, and rollback keep any single change small and reversible.

## 7. The fair metric: bits per character

A bigger vocabulary makes fewer tokens, which makes loss per token look different even if the model did not change. Always compare vocabularies with:

```
bpc = ( total negative log-likelihood in nats over a text ) / ( ln(2) * number of characters in that text )
```

Count characters (or bytes) in the original text, not tokens. Use the same held-out text for every comparison.

## 8. Fairness test for under-served languages

General-purpose tokenizers are often trained mostly on English, which can split text in other languages (especially with diacritics and tone marks) into many small pieces.

Plan:

1. Collect small held-out text samples in English, Yoruba, Igbo, and Hausa. Choose licensed or self-written text and check licenses first.
2. For each language measure **fertility**: average tokens per word, and average tokens per character.
3. Compare: an off-the-shelf byte-level BPE vs UBM v0 vs UBM v1 trained on a corpus that includes those languages.
4. Report fertility ratios relative to English. A big gap is the thing UBM can reduce by including these languages in training and promotion.

No numbers are claimed here; this is an experiment to run.

## 9. Interface

```
class UBM:
    vocab_version: int
    def encode(text: str) -> list[int]
    def decode(ids: list[int]) -> str
    def expand(id: int) -> bytes                   # bytes of one base
    def parts(id: int) -> tuple[int, int] | None
    def add_bases(records: list[BaseRecord]) -> list[int]
    def tombstone(id: int) -> None
    def save(path) / load(path)
```

Vocabulary file (JSON): header with `vocab_version`, `normalization`, `pretokenizer` regex, then a list of base records.

## 10. Edge cases

| Case | Handling |
|---|---|
| Emoji, rare scripts | Always representable as bytes |
| Invalid UTF-8 from model output | Replacement character on decode |
| Very long words | Merges stop at a maximum base length (default 32 bytes) |
| Numbers | Pre-tokenizer splits digits into short groups (default up to 3 digits) so numbers stay compositional |
| Whitespace runs | Own chunk type; deterministic |
| Different normalization forms in training data | Normalize everything to NFC first |
| Rolled-back base | Tombstone, keep the ID reserved |

## 11. Risks

| Risk | Mitigation |
|---|---|
| Vocabulary drift breaks trained dots | Checkpoint-only batches, settling phase, rollback |
| Promotion rule rewards memorizable junk | Rank by agreement and confidence together, rollback test on held-out text |
| Tokenizer becomes slower than the model | Cache encodings, implement merges with a priority structure |
| Claimed novelty is weaker than thought | Treat UBM v0 as BPE; claim only what the v1 experiments show |

## 12. Tests

1. Round trip: `decode(encode(x)) == normalize(x)` over random Unicode strings, including combining marks and emoji.
2. No unknowns: every string encodes without error.
3. Determinism: same text and same vocabulary give identical IDs.
4. ID stability: after adding bases, previously issued IDs keep meaning the same bytes.
5. Expansion: `expand(id)` equals the concatenation of `expand(left)` and `expand(right)` for every merged base.
6. Rollback: after tombstoning a base, text encodes as it did before that base existed.
7. Cross-check v0 against an established byte-level BPE library on the same corpus (merge lists should match for the same training rules, up to tie-breaking).
