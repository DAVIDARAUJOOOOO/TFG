import Levenshtein
import jiwer


def compute_metrics(gt, text, duration):
    ref_gt = gt.split()
    ref_text = text.split()

    return {
        "cer": Levenshtein.distance(gt, text) / len(gt),
        "wer": jiwer.wer(gt, text),
        "velocitat": len(ref_text) / duration,
        "ratio": len(ref_text) / len(ref_gt),
    }
