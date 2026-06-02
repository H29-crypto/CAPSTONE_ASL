"""
smoke_test.py - CorrNet + MSLR pipeline smoke test
---------------------------------------------------
Verifies forward passes, shape correctness, padding math, KD loss,
and gloss-to-text conversion - all without GPU or internet access.

Usage:
    cd corrnet
    python smoke_test.py             # fast checks (CPU, <30 s)
    python smoke_test.py --corrnet   # also run 3D-ResNet forward pass
                                     #   (downloads resnet18 ImageNet weights once)

Exit code 0 = all checks passed.
"""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CORRNET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CORRNET_DIR))

PASS = "[PASS]"
FAIL = "[FAIL]"
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    if condition:
        print(f"  {PASS}  {name}")
    else:
        msg = f"{name}" + (f"  [{detail}]" if detail else "")
        print(f"  {FAIL}  {name}" + (f"  -- {detail}" if detail else ""))
        failures.append(msg)
    return condition


# =============================================================================
# 1. MSLRStudent - instantiation & forward pass
# =============================================================================

def test_mslr_student(gloss_dict: dict):
    print("\n--- MSLRStudent -----------------------------------------------")
    from mslr_student import MSLRStudent

    model = MSLRStudent(gloss_dict)
    n_params = sum(p.numel() for p in model.parameters())
    check("MSLRStudent instantiation", True)
    check("param count < 10 M", n_params < 10_000_000, f"{n_params:,}")

    # Training mode forward
    B, T, D = 2, 32, 444
    x   = torch.randn(B, T, D)
    lgt = torch.tensor([T, T - 4], dtype=torch.long)

    model.train()
    out = model(x, lgt)

    expected_keys = {"sequence_logits", "conv_logits", "feat_len", "recognized_sents"}
    check("output keys present", expected_keys.issubset(out.keys()),
          str(set(out.keys())))

    seq_logits  = out["sequence_logits"]   # (T', B, C)
    conv_logits = out["conv_logits"]
    feat_len    = out["feat_len"]

    check("sequence_logits ndim = 3",  seq_logits.ndim == 3,
          f"{seq_logits.shape}")
    check("seq B dim matches",  seq_logits.shape[1] == B,
          f"got {seq_logits.shape[1]}")
    check("num_classes correct", seq_logits.shape[2] == len(gloss_dict) + 1,
          f"got {seq_logits.shape[2]}")
    check("conv_logits shape matches seq",
          conv_logits.shape == seq_logits.shape,
          f"{conv_logits.shape} vs {seq_logits.shape}")
    check("feat_len shape = (B,)", feat_len.shape == (B,),
          f"{feat_len.shape}")
    check("feat_len[0] >= feat_len[1]",
          feat_len[0].item() >= feat_len[1].item(),
          f"{feat_len.tolist()}")
    check("recognized_sents is None during training",
          out["recognized_sents"] is None)

    # Eval mode - decoder should run
    model.eval()
    with torch.no_grad():
        out_eval = model(x, lgt)
    check("recognized_sents is list during eval",
          isinstance(out_eval["recognized_sents"], list))
    check("recognized_sents length = B",
          len(out_eval["recognized_sents"]) == B,
          f"got {len(out_eval['recognized_sents'])}")

    # predict() convenience API
    glosses, confs = model.predict(x[:1], lgt[:1])
    check("predict() returns two lists", isinstance(glosses, list) and isinstance(confs, list))

    # CTC loss backward
    model.train()
    ctc_fn = torch.nn.CTCLoss(reduction="none", zero_infinity=True)
    log_p  = F.log_softmax(seq_logits, dim=-1)
    labels = torch.randint(1, len(gloss_dict) + 1, (B,))
    l_lens = torch.ones(B, dtype=torch.long)
    ctc    = ctc_fn(log_p, labels, feat_len.int(), l_lens).mean()
    check("CTC loss is finite", torch.isfinite(ctc).item(), f"loss={ctc.item():.4f}")
    ctc.backward()
    check("backward() succeeds without error", True)

    # Save / load round-trip
    model.eval()
    with torch.no_grad():
        ref_out = model(x, lgt)
    ref_logits = ref_out["sequence_logits"].clone()

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = f"{td}/student.pt"
        model.save(ckpt_path)
        check("save() writes file", Path(ckpt_path).exists())

        loaded = MSLRStudent.load(ckpt_path, gloss_dict)
        loaded.eval()
        with torch.no_grad():
            loaded_out = loaded(x, lgt)
        loaded_logits = loaded_out["sequence_logits"]

        max_diff = (ref_logits - loaded_logits).abs().max().item()
        check("load() weights identical to saved", max_diff == 0.0,
              f"max diff={max_diff:.2e}")

    return model


# =============================================================================
# 2. pad_and_make_batch padding math
# =============================================================================

def test_padding_math():
    print("\n--- pad_and_make_batch & padding math -------------------------")
    from corrnet_webcam import pad_and_make_batch, LEFT_PAD, TOTAL_STRIDE

    check("LEFT_PAD = 6",        LEFT_PAD == 6,        f"got {LEFT_PAD}")
    check("TOTAL_STRIDE = 4",    TOTAL_STRIDE == 4,    f"got {TOTAL_STRIDE}")

    # Synthetic frames: random 256x256 RGB images (dtype uint8)
    T = 30
    frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(T)]

    vid, lgt = pad_and_make_batch(frames)
    expected_vid_len = math.ceil(T / TOTAL_STRIDE) * TOTAL_STRIDE + 2 * LEFT_PAD

    check("vid shape (1, vid_len, 3, 224, 224)",
          vid.shape == (1, expected_vid_len, 3, 224, 224),
          f"{vid.shape}")
    check("lgt = [vid_len]",
          lgt.tolist() == [expected_vid_len],
          f"{lgt.tolist()} expected [{expected_vid_len}]")
    check("pixel range [-1, 1]",
          vid.min().item() >= -1.0 - 1e-5 and vid.max().item() <= 1.0 + 1e-5,
          f"min={vid.min().item():.3f} max={vid.max().item():.3f}")

    # Verify feat_len formula: after K5-P2-K5-P2 output length equals ceil(T/4)
    from modules.tconv import TemporalConv
    tconv    = TemporalConv(input_size=512, hidden_size=512, conv_type=2, num_classes=10)
    feat_len = tconv.update_lgt(lgt)
    expected_feat_len = math.ceil(T / TOTAL_STRIDE)
    check("feat_len = ceil(T/4)",
          feat_len.item() == expected_feat_len,
          f"got {feat_len.item()} expected {expected_feat_len}")


# =============================================================================
# 3. KD loss (from train_mslr.py)
# =============================================================================

def test_kd_loss():
    print("\n--- KD loss ---------------------------------------------------")
    from train_mslr import kd_loss

    B, T_s, T_t, C = 3, 12, 10, 100
    student_logits  = torch.randn(T_s, B, C)
    feat_len        = torch.tensor([T_s, T_s - 2, T_s - 4], dtype=torch.long)
    teacher_logits  = [torch.randn(T_t, C) for _ in range(B)]

    loss = kd_loss(student_logits, teacher_logits, feat_len, tau=4.0)
    check("kd_loss returns scalar",  loss.ndim == 0)
    check("kd_loss is finite",       torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
    check("kd_loss >= 0",            loss.item() >= 0.0)

    # Identical distributions => KL = 0
    t_logits_same = [student_logits[:T_t, 0].detach() for _ in range(B)]
    loss_same = kd_loss(student_logits, t_logits_same, feat_len, tau=4.0)
    check("KL(p||p) ~ 0",           abs(loss_same.item()) < 0.1,
          f"loss={loss_same.item():.6f}")


# =============================================================================
# 4. PseudoDataset + collate (train_mslr.py)
# =============================================================================

def test_pseudo_dataset(gloss_dict: dict):
    print("\n--- PseudoDataset + collate -----------------------------------")
    from train_mslr import PseudoDataset, collate
    from torch.utils.data import DataLoader

    # Write two synthetic .npz files to a temp dir
    T_feat, T_teach, C = 50, 12, len(gloss_dict) + 1
    with tempfile.TemporaryDirectory() as td:
        for i in range(3):
            np.savez(
                f"{td}/REGEN_{i:03d}.npz",
                features       = np.random.randn(T_feat, 444).astype(np.float32),
                teacher_logits = np.random.randn(T_teach, C).astype(np.float32),
                hard_label     = np.array(1, dtype=np.int64),
                gloss          = "REGEN",
                feat_len       = np.array(T_teach, dtype=np.int64),
            )

        paths = sorted(Path(td).glob("*.npz"))
        ds    = PseudoDataset(paths, do_augment=False)
        check("PseudoDataset len = 3", len(ds) == 3, f"got {len(ds)}")

        sample = ds[0]
        check("sample has 4 elements", len(sample) == 4)
        X, T_soft, y, L = sample
        check("X is float tensor",      X.dtype == torch.float32)
        check("teacher logit is tensor", isinstance(T_soft, torch.Tensor))
        check("hard label is int",       isinstance(y, int))
        check("length = T_feat",         L == T_feat, f"got {L}")

        dl    = DataLoader(ds, batch_size=2, collate_fn=collate)
        batch = next(iter(dl))
        feats, lens, t_logs, labels, l_lens = batch
        check("feats shape (B, T_max, 444)", feats.ndim == 3 and feats.shape[2] == 444,
              f"{feats.shape}")
        check("lens is LongTensor",          lens.dtype == torch.long)
        check("t_logs is list of tensors",
              isinstance(t_logs, list) and isinstance(t_logs[0], torch.Tensor))
        check("labels shape = (B,)",         labels.shape == (2,), f"{labels.shape}")
        check("l_lens all ones",             (l_lens == 1).all().item())


# =============================================================================
# 5. gloss_to_text
# =============================================================================

def test_gloss_to_text():
    print("\n--- gloss_to_text ---------------------------------------------")
    from gloss_to_text import glosses_to_german

    cases = [
        (["HEUTE", "REGEN", "KOMMEN"],    "Heute Regen kommt."),
        (["SONNE", "WARM"],               "Sonne warm."),
        (["loc-NORD"],                     "Im Norden."),
        (["__PU__", "IX"],                ""),
    ]
    for glosses, expected in cases:
        result = glosses_to_german(glosses)
        ok = (result == "") if expected == "" else (result == expected)
        check(f"glosses_to_german {glosses[:2]}", ok,
              f"got '{result}' expected '{expected}'")

    # Number translations
    num_cases = [
        (["MINUS", "ZWEI", "GRAD"],            "Minus zwei Grad."),
        (["ZEHN", "BIS", "FUENFZEHN", "GRAD"], "Zehn bis fünfzehn Grad."),
        (["MAXIMAL", "ZWANZIG", "GRAD"],        "Maximal zwanzig Grad."),
        (["NULL", "GRAD"],                      "Null Grad."),
        (["SIEBEN", "ACHT", "NEUN"],            "Sieben acht neun."),
        (["ELF", "BIS", "DREIZEHN", "GRAD"],   "Elf bis dreizehn Grad."),
    ]
    for glosses, expected in num_cases:
        result = glosses_to_german(glosses)
        check(f"numbers {glosses[:2]}", result == expected,
              f"got '{result}' expected '{expected}'")

    # Should not crash on unknown tokens
    try:
        glosses_to_german(["UNKNOWN_TOKEN", "GRAD", "MINUS"])
        check("no crash on unknown tokens", True)
    except Exception as e:
        check("no crash on unknown tokens", False, str(e))


# =============================================================================
# 6. Optional: SLRModel (CorrNet 3D-ResNet) forward pass
# =============================================================================

def test_corrnet_slrmodel(gloss_dict: dict):
    print("\n--- SLRModel (CorrNet 3D-ResNet18) ----------------------------")
    print("   NOTE: resnet18() downloads ~45 MB ImageNet weights on first run")

    from corrnet_webcam import load_model, pad_and_make_batch, run_inference

    if not CKPT_PATH.exists():
        print(f"  [WARN]  checkpoint not found: {CKPT_PATH}")
        print("     skipping SLRModel test -- download from project README")
        return

    print(f"  Loading from {CKPT_PATH.name} ...")
    model = load_model(CKPT_PATH, gloss_dict)
    n_params = sum(p.numel() for p in model.parameters())
    check("SLRModel loaded",          True)
    check("param count > 10 M",       n_params > 10_000_000, f"{n_params:,}")

    # Dry-run inference with synthetic frames (T=25, no motion so may return [])
    T = 25
    frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(T)]
    device = torch.device("cpu")

    glosses, confs = run_inference(model, frames, device)
    check("run_inference returns two lists",
          isinstance(glosses, list) and isinstance(confs, list))
    check("confs and glosses same length",
          len(glosses) == len(confs))
    if glosses:
        check("glosses are strings",   all(isinstance(g, str) for g in glosses))
        check("confs in [0, 1]",       all(0.0 <= c <= 1.0 for c in confs))


# =============================================================================
# Main
# =============================================================================

CKPT_PATH = CORRNET_DIR / "weights" / "corrnet_phoenix2014T.pt"
DICT_PATH = CORRNET_DIR / "preprocess" / "phoenix2014-T" / "gloss_dict.npy"


def main():
    ap = argparse.ArgumentParser(description="CorrNet / MSLR smoke tests")
    ap.add_argument("--corrnet", action="store_true",
                    help="Also run SLRModel (3D ResNet-18) forward pass test")
    args = ap.parse_args()

    print("=" * 60)
    print("  CorrNet + MSLR smoke test")
    print("=" * 60)

    if not DICT_PATH.exists():
        sys.exit(f"\n[ERROR] Gloss dict not found: {DICT_PATH}\n")
    gloss_dict = np.load(str(DICT_PATH), allow_pickle=True).item()
    print(f"\n  Vocabulary: {len(gloss_dict)} Phoenix glosses loaded")

    test_mslr_student(gloss_dict)
    test_padding_math()
    test_kd_loss()
    test_pseudo_dataset(gloss_dict)
    test_gloss_to_text()

    if args.corrnet:
        test_corrnet_slrmodel(gloss_dict)

    # Summary
    print("\n" + "=" * 60)
    if not failures:
        print("  ALL CHECKS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"    - {f}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
