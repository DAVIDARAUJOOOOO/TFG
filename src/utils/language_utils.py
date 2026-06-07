import re


def obtenir_llengua(l1_match, mapa_llengues):
    l1 = l1_match.group(1) if l1_match else "No especificat"
    l1 = l1.strip("_")
    l1 = l1.replace("_i_", "_").replace("_y_", "_")
    l1 = mapa_llengues.get(l1, l1)
    return l1
