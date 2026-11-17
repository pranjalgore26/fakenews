import re


def clean(text: str) -> str:
    """Normalise article text before training and prediction.

    Removes agency datelines such as "WASHINGTON (Reuters) -" and URLs.
    Without this, the model learns to spot the dateline instead of the
    writing style, and scores look much better than they really are.
    """
    text = str(text)
    text = re.sub(r"^.{0,60}?\(Reuters\)\s*-?\s*", "", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
