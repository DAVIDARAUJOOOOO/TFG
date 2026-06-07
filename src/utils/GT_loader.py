from utils.text_cleaning import neteja


def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        gt_brut = f.read()

    return neteja(gt_brut)