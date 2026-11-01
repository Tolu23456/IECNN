"""Tests for UBM (spec 07, section 12). All originally-skipped stubs implemented."""
import os
import random
import tempfile

from UBM import UBM, BaseRecord, FIRST_LEARNED_ID, normalize_text
from UBM.tokenizer import chunk_text
from UBM.train_bpe import train_ubm_v0, ubm_from_merges


def _demo_text():
    return (
        "the cat sat on the mat. the cat ate a rat. THE END of the story. "
        "numbers 123 456 here; punctuation! words: dog, dogs, dogged."
    )


def _trained_ubm(text: str, target_vocab_size: int = 280):
    """Trained tokenizer + its merge records (spec 07 file, plain wrapper)."""
    return train_ubm_v0(text, target_vocab_size=target_vocab_size)


# ---------------------------------------------- independent BPE reference --
def _reference_merge_byte_strings(text: str, target_vocab_size: int) -> list[bytes]:
    """Naive BPE reference that recounts everything each round.

    Own chunking/normalization/tie-break/cap must match the trainer exactly:
    same pre-tokenizer, most frequent pair first, ties resolved by the smallest
    (left, right) pair, merged forms longer than 32 bytes are banned, stop at
    target size or when no pair occurs twice.
    """
    chunks = [list(c.encode("utf-8")) for c in chunk_text(normalize_text(text)) if c]
    bytes_of = {i: bytes([i]) for i in range(256)}
    merges: list[bytes] = []
    banned: set[tuple[int, int]] = set()

    # Number of learned bases needed so the total vocabulary hits the target.
    needed = max(0, target_vocab_size - (256 + 8))
    while len(merges) < needed:
        counts: dict[tuple[int, int], int] = {}
        for ch in chunks:
            for i in range(len(ch) - 1):
                pair = (ch[i], ch[i + 1])
                if pair in banned:
                    continue
                counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        best_count = max(counts.values())
        if best_count < 2:
            break
        best = min(p for p, c in counts.items() if c == best_count)
        left, right = best
        merged = bytes_of[left] + bytes_of[right]
        if len(merged) > 32:                     # mirror the trainer's length cap
            banned.add(best)
            continue
        new_id = 10000 + len(merges)             # reference-internal id space
        for c_i, ch in enumerate(chunks):
            out: list[int] = []
            i = 0
            while i < len(ch):
                if i + 1 < len(ch) and (ch[i], ch[i + 1]) == best:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ch[i])
                    i += 1
            chunks[c_i] = out
        bytes_of[new_id] = merged
        merges.append(merged)
    return merges


# ------------------------------------------------------------------- tests --
def test_vocabulary_layout_constants():
    ubm = UBM()
    assert len(ubm.bases) == 264
    assert ubm.bases[0].bytes == b"\x00"
    assert ubm.bases[255].bytes == b"\xff"
    assert ubm.bases[256].bytes == b""          # specials have no byte payload
    assert FIRST_LEARNED_ID == 264


def test_round_trip_random_unicode_with_combining_marks_and_emoji():
    ubm, _ = _trained_ubm(_demo_text())
    samples = [
        "ọ̀", "ẹ́", "ṣ", "é", "e\u0301\u0327",       # Yoruba-style decomposed sequences
        "👩\u200d🚀", "🎉🚀🕌", "Ångström", "日本語テキスト", "Здравствуй!",
    ]
    rng = random.Random(7)
    alphabet = "abcXYZ \n\t.,!?" + "\u0301\u0300\u0323"
    for _ in range(300):
        samples.append("".join(rng.choice(alphabet) for _ in range(rng.randint(0, 50))))
    for s in samples:
        ids = ubm.encode(s)
        assert ubm.decode(ids) == normalize_text(s), f"round trip failed for {s!r}"


def test_every_string_encodes_without_error():
    ubm, _ = _trained_ubm(_demo_text())
    rng = random.Random(11)
    for _ in range(200):
        s = "".join(chr(rng.randrange(0x20, 0x2FFF)) for _ in range(rng.randint(0, 40)))
        ids = ubm.encode(s)                     # must not raise
        assert isinstance(ids, list)
    for s in ["\x00\x01\x1a\x7f", "\uE000\uE001", "\ufffd", ""]:
        ubm.encode(s)


def test_determinism_same_text_same_ids():
    ubm, _ = _trained_ubm(_demo_text())
    text = "the cat sat on the mat; 123 numbers! ọ̀ emoji 🚀"
    assert ubm.encode(text) == ubm.encode(text)
    _, merges_a = _trained_ubm(_demo_text())
    _, merges_b = _trained_ubm(_demo_text())
    assert [m.bytes for m in merges_a] == [m.bytes for m in merges_b]


def test_id_stability_after_adding_bases():
    ubm, _ = _trained_ubm(_demo_text())
    text = "the cat sat"
    before = list(ubm.encode(text))
    expansions = [ubm.expand(i) for i in range(len(ubm.bases))]
    n = len(ubm.bases)
    ubm.add_bases([BaseRecord(id=n, bytes=b"zz", parts=(122, 122), origin="promotion")])
    assert ubm.bases[n].bytes == b"zz"
    assert ubm.encode(text) == before           # adding a base does not rewrite old text
    for i, bs in enumerate(expansions):
        assert ubm.expand(i) == bs              # every old id keeps its bytes


def test_expand_equals_concat_of_parts_for_every_merged_base():
    ubm, merges = _trained_ubm(_demo_text())
    for rec in merges:
        left, right = rec.parts
        assert ubm.expand(rec.id) == ubm.expand(left) + ubm.expand(right)


def test_rollback_after_tombstone_restores_old_encoding():
    _, merges = _trained_ubm(_demo_text())
    text = "the cat sat on the mat."
    byte_only = UBM().encode(text)
    ubm, _ = _trained_ubm(_demo_text())
    with_merges = ubm.encode(text)
    assert with_merges != byte_only and len(with_merges) < len(byte_only)
    # Tombstoning base X must return the encoding to the state that existed
    # before X was created (spec 07, section 6.2): the same as a tokenizer with
    # only the earlier merges.
    for idx in range(len(merges) - 1, -1, -1):
        ubm.tombstone(merges[idx].id)
        expected = ubm_from_merges(merges, idx).encode(text)
        got = ubm.encode(text)
        assert got == expected, (
            f"rollback failed after tombstoning {merges[idx].id}: "
            f"got {got[:10]} wanted {expected[:10]}"
        )
    assert got == byte_only, "with every merge inactive the encoding is plain bytes"
    for rec in merges:
        ubm.revive(rec.id)
    assert ubm.encode(text) == with_merges


def test_v0_matches_reference_bpe_on_same_corpus():
    text = _demo_text()
    target = 288
    ubm, merges = _trained_ubm(text, target_vocab_size=target)
    ours = [ubm.expand(m.id) for m in merges]
    ref = _reference_merge_byte_strings(text, target_vocab_size=target)
    assert ours == ref


def test_decode_replaces_invalid_utf8():
    ubm, _ = _trained_ubm(_demo_text())
    assert ubm.decode([0x80]) == "\ufffd"       # lone continuation byte


def test_save_load_round_trip():
    ubm, _merges = _trained_ubm(_demo_text())
    text = "the cat ate a rat; the end."
    ids_before = ubm.encode(text)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "vocab.json")
        ubm.save(path)
        loaded = UBM.load(path)
        assert loaded.encode(text) == ids_before
        assert len(loaded.bases) == len(ubm.bases)
        for a, b in zip(loaded.bases, ubm.bases):
            assert a.bytes == b.bytes and a.parts == b.parts and a.active == b.active
