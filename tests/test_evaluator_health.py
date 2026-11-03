"""Evaluator health checks (Tolu's Checkpoint-B follow-up list, item 6):
1. at step 0 (fresh init) the evaluator must return ~uniform bpc;
2. on a text slice the model was trained on, it must return LOW bpc.
Together these catch the classic silent breaks: off-by-one shifting, wrong
char accounting, vocab mismatch, or a mode where predictions are shuffled."""
import math

import torch

from train.evaluate import evaluate_bpc, bpc_from_nll


def _tiny_ubm():
    """64 single-byte 'characters' as bases (same idea as test_training)."""
    from UBM import UBM, BaseRecord
    ubm = UBM()
    recs = [BaseRecord(id=len(ubm.bases) + i, bytes=ch.encode())
            for i, ch in enumerate(
                "abcdefghijklmnopqrstuvwxyz_ABCDEFGHIJKLMNOPQRSTUVWXYZ")]
    ubm.add_bases(recs)
    return ubm


def test_evaluator_uniform_at_init_and_low_on_trained_text():
    """Fresh model: bpc ~ uniform; same architecture trained on repeated
    structure: bpc far below uniform on that training slice."""
    ubm = _tiny_ubm()
    vocab = len(ubm.bases)

    # deterministic repetitive training text (~75 tokens; tiny enough that the
    # 2-core dev box learns it in seconds, big enough to beat uniform clearly)
    text = "abcdef abcdef abcdef " * 4
    ids_text = ubm.encode(text)
    assert len(ids_text) > 40

    from models.baselines import GRULanguageModel
    torch.manual_seed(3)
    fresh = GRULanguageModel(vocab, d=32, hidden=64)
    fresh.eval()

    bpc_uniform = evaluate_bpc(fresh, ubm, text, seq_len=160)
    # uniform expectation: log2(V) bits/predicted token, 1 char per token here
    expected_uniform = math.log2(vocab)
    assert abs(bpc_uniform - expected_uniform) < 0.35, bpc_uniform

    # train the same architecture until the repeated text is learned
    torch.manual_seed(3)
    trained = GRULanguageModel(vocab, d=16, hidden=32)
    opt = torch.optim.AdamW(trained.parameters(), lr=3e-3)
    x = torch.tensor([ids_text[:-1]])
    y = torch.tensor([ids_text[1:]])
    trained.train()
    for _ in range(60):
        out = trained(x)
        loss = torch.nn.functional.nll_loss(
            out["final_log_probs"].reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    trained.eval()
    bpc_train = evaluate_bpc(trained, ubm, text, seq_len=160)

    assert bpc_train < 0.55 * bpc_uniform, (bpc_train, bpc_uniform)


def test_evaluator_full_text_char_identity():
    """On a mixed normal text the char denominator must equal the text's
    length (spec 07 section 7 accounting identity, end-to-end)."""
    text = ("Emma was delightfully plain nineteen times; tables, mirrors "
            "and windows agreed.\n\tShe said nobody truly\n") * 40
    ubm = _tiny_ubm()
    from models.baselines import GRULanguageModel
    torch.manual_seed(5)
    model = GRULanguageModel(len(ubm.bases), d=16, hidden=32)
    model.eval()
    ids = ubm.encode(text)
    # accounting kernel: predicted tokens' expansion lengths count the chars.
    # The corpus UBM guarantees len(text) exactly; the tools-only synthetic UBM
    # folds the trailing newline into the last token, so allow the ±1 boundary.
    chars_predicted = sum(len(ubm.expand(int(i))) for i in ids[1:])
    assert abs(chars_predicted - len(text)) <= 1, (chars_predicted, len(text))
    bpc = evaluate_bpc(model, ubm, text, seq_len=128)
    assert 0.0 < bpc < 20.0
    # bpc arithmetic hand-check on an exactly-uniform model
    class _Uniform(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))   # so to() works
        def forward(self, x, **kw):
            v = id(self.p)          # any int; real vocab size read from ubm
            n = _V
            lp = torch.zeros(*x.shape, n)
            return {"final_log_probs": torch.log_softmax(lp, -1)}
    _V = len(ubm.bases)
    uniform = _Uniform()
    nll_h = (len(ids) - 1) * math.log(_V)
    assert math.isclose(evaluate_bpc(uniform, ubm, text, seq_len=128),
                        bpc_from_nll(nll_h, chars_predicted), rel_tol=1e-6)