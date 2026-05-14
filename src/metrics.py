
def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a 
    bx1, by1, bx2, by2 = box_b 

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h 

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area 

    return inter_area / union if union > 0 else 0.0 


def max_score(predictions): 
    if not predictions: 
        return 0.0 
    return max(pred["score"] for pred in predictions)


def false_situation_activation(prediction, threshold=0.3):
    return max_score(prediction["predictions"]) >= threshold


def situation_discrimination_score(dangerous_score, nondangerous_score):
    return dangerous_score - nondangerous_score


def prompt_specificity_gap(situation_score, composition_score):
    return situation_score - composition_score


def is_false_situation_activation(expected: str, score: float, threshold: float) -> bool:
    return expected == "negative" and score >= threshold