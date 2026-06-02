"""
decoder.py — CTC greedy decoding.
"""

import torch


def greedy_ctc_decode(
    log_probs: torch.Tensor,
    blank_idx: int = 1,
) -> list[int]:
    """
    Greedy CTC decoding: argmax over vocabulary → collapse repeats → remove blanks.

    Args:
        log_probs: (T, vocab_size) log-softmax output for a single sequence.
        blank_idx: index of the CTC blank token (default 1).

    Returns:
        List of integer token indices with consecutive repeats collapsed
        and blank tokens removed.

    Example:
        Input argmax path:  [5, 5, 1, 3, 3, 1, 5]   (blank=1)
        After collapse:     [5, 1, 3, 1, 5]
        After blank removal:[5, 3, 5]
    """
    best_path = log_probs.argmax(dim=-1).tolist()   # (T,) int list

    decoded: list[int] = []
    previous_token: int | None = None

    for token in best_path:
        if token != previous_token:
            if token != blank_idx:
                decoded.append(token)
        previous_token = token

    return decoded
