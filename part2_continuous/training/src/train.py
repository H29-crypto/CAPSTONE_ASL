"""Training engine for the frozen plan.

One entry point covers the whole experiment table:

    run_training(cfg, streams=("rgb","kp"), run_name="E3_concat", seed=42)

Ablations are config overrides, so all rows share the same schedule
(matched-conditions rule), e.g.:

    run_training(cfg, ("rgb","kp"), "E4_aux", overrides={"model.aux_ctc": True})
    run_training(cfg, ("rgb","kp"), "A2_transformer",
                 overrides={"model.aux_ctc": True, "model.head": "transformer"})

Resumable: re-running the same run_name continues from last.pt.
"""
import copy
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import FeatureDataset, collate
from .decode import evaluate_greedy
from .models import CSLRModel, count_params
from .utils import get_device, load_json, p, save_json, set_seed
from .vocab import load_vocab


def apply_overrides(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    for key, val in (overrides or {}).items():
        d = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            d = d[k]
        d[parts[-1]] = val
    return cfg


def load_manifest(cfg, split):
    return load_json(p(cfg, "manifests") / f"{split}.json")


def make_loaders(cfg, streams, gloss2id, smoke=False):
    tr_items = load_manifest(cfg, "train")
    dv_items = load_manifest(cfg, "dev")
    tr_ids = dv_ids = None
    if smoke:
        tr_ids = load_json(p(cfg, "manifests") / "smoke_train_ids.json")
        dv_ids = load_json(p(cfg, "manifests") / "smoke_dev_ids.json")
    tr = FeatureDataset(cfg, tr_items, "train", streams, gloss2id,
                        train_mode=True, ids_subset=tr_ids)
    dv = FeatureDataset(cfg, dv_items, "dev", streams, gloss2id,
                        train_mode=False, ids_subset=dv_ids)
    t = cfg["train"]
    tr_loader = DataLoader(tr, batch_size=t["batch_size"], shuffle=True,
                           num_workers=t["num_workers"], collate_fn=collate,
                           drop_last=True)
    dv_loader = DataLoader(dv, batch_size=t["batch_size"], shuffle=False,
                           num_workers=t["num_workers"], collate_fn=collate)
    return tr_loader, dv_loader


def _combined_loss(out, batch, cfg, criterion, device):
    m = cfg["model"]
    tgt = batch["targets"].to(device)
    tlen = batch["target_lengths"].to(device)
    loss = criterion(out["main"], tgt, out["out_lengths"], tlen)
    if "aux_rgb" in out:
        loss = loss + m["aux_weight_rgb"] * criterion(out["aux_rgb"], tgt,
                                                      out["out_lengths"], tlen)
    if "aux_kp" in out:
        loss = loss + m["aux_weight_kp"] * criterion(out["aux_kp"], tgt,
                                                     out["out_lengths"], tlen)
    return loss


def run_training(cfg, streams, run_name, seed=42, smoke=False, overrides=None,
                 max_epochs=None, resume=True):
    cfg = apply_overrides(cfg, overrides)
    set_seed(seed)
    device = get_device()

    gloss2id, id2gloss = load_vocab(p(cfg, "manifests") / "vocab.json")
    num_classes = len(gloss2id) + 1                       # + blank at index 0
    tr_loader, dv_loader = make_loaders(cfg, streams, gloss2id, smoke)

    model = CSLRModel(cfg, num_classes, streams).to(device)
    t = cfg["train"]
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"],
                           weight_decay=t["weight_decay"])
    epochs = max_epochs or t["epochs"]
    warm = t["warmup_epochs"]

    def lr_lambda(ep):
        if ep < warm:
            return (ep + 1) / max(warm, 1)
        prog = (ep - warm) / max(epochs - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    criterion = nn.CTCLoss(blank=0, zero_infinity=t["ctc_zero_infinity"])

    run_dir = p(cfg, "runs") / f"{run_name}_seed{seed}{'_SMOKE' if smoke else ''}"
    run_dir.mkdir(parents=True, exist_ok=True)
    hist_path, last_path, best_path = (run_dir / "history.json",
                                       run_dir / "last.pt", run_dir / "best.pt")
    history, start_ep, best_wer = [], 0, float("inf")
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep, best_wer = ck["epoch"] + 1, ck["best_wer"]
        history = load_json(hist_path) if hist_path.exists() else []
        print(f"[{run_name}] resumed at epoch {start_ep} (best dev WER {best_wer:.2f})")

    print(f"[{run_name}] streams={streams} params={count_params(model)/1e6:.2f}M "
          f"device={device} smoke={smoke}")

    for ep in range(start_ep, epochs):
        model.train()
        t0, total, nb = time.time(), 0.0, 0
        for i, batch in enumerate(tr_loader):
            out = model(batch, device)
            loss = _combined_loss(out, batch, cfg, criterion, device)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip_norm"])
            opt.step()
            total += loss.item(); nb += 1
            if (i + 1) % t["log_every"] == 0:
                print(f"  ep{ep} it{i+1}/{len(tr_loader)} loss {total/nb:.3f}")
        sched.step()

        dev, _, _ = evaluate_greedy(model, dv_loader, id2gloss, device)
        rec = {"epoch": ep, "train_loss": total / max(nb, 1),
               "dev_wer": dev["wer"], "dev_sub": dev["sub"], "dev_ins": dev["ins"],
               "dev_del": dev["del"], "lr": sched.get_last_lr()[0],
               "minutes": (time.time() - t0) / 60}
        history.append(rec); save_json(history, hist_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best_wer": best_wer,
                    "cfg": cfg, "streams": streams, "seed": seed}, last_path)
        flag = ""
        if dev["wer"] < best_wer:
            best_wer = dev["wer"]
            torch.save({"model": model.state_dict(), "cfg": cfg,
                        "streams": streams, "seed": seed, "epoch": ep,
                        "dev_wer": best_wer}, best_path)
            flag = "  <-- new best"
        print(f"[{run_name}] ep{ep}: loss {rec['train_loss']:.3f} | "
              f"dev WER {dev['wer']:.2f} (S{dev['sub']} I{dev['ins']} D{dev['del']})"
              f" | {rec['minutes']:.1f} min{flag}")

    print(f"[{run_name}] done. best dev WER {best_wer:.2f} -> {best_path}")
    return {"run_name": run_name, "seed": seed, "best_dev_wer": best_wer,
            "best_path": str(best_path), "history": history}


def load_model_from_ckpt(ckpt_path, device=None):
    device = device or get_device()
    ck = torch.load(ckpt_path, map_location=device)
    g2i, i2g = load_vocab(p(ck["cfg"], "manifests") / "vocab.json")
    model = CSLRModel(ck["cfg"], len(g2i) + 1, ck["streams"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["cfg"], i2g
