from collections import defaultdict


def calc_hierarchical_metrics(preds, golds):
    def calc_p_r_f1_from_tp_fp_fn(tp, fp, fn):
        try:
            p = tp / (tp + fp)
        except ZeroDivisionError:
            p = 0.0
        try:
            r = tp / (tp + fn)
        except ZeroDivisionError:
            r = 0.0
        try:
            f1 = 2 * p * r / (p + r)
        except ZeroDivisionError:
            f1 = 0.0
        return {"p": p, "r": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    heading_result = {"tp": 0, "fp": 0, "fn": 0}
    text_result = {"tp": 0, "fp": 0, "fn": 0}
    overall_result = {"tp": 0, "fp": 0, "fn": 0}
    pred_nodes = set()
    gold_nodes = set()

    final_metrics = {}
    for pred, gold in zip(preds, golds):
        pred_level2heading = defaultdict(set)
        pred_level2text = defaultdict(set)
        pred_level2overall = defaultdict(set)
        gold_level2heading = defaultdict(set)
        gold_level2text = defaultdict(set)
        gold_level2overall = defaultdict(set)

        for node in pred:
            label = node["label"]
            if label == "Root":
                continue
            level = node["depth"]
            pred_nodes.add((level, label, node["content"]))
            pred_level2overall[level].add((label, node["content"]))
            if label == "Heading":
                pred_level2heading[level].add(node["content"])
            else:
                pred_level2text[level].add(node["content"])

        for node in gold:
            label = node["label"]
            if label == "Root":
                continue
            level = node["depth"]
            gold_nodes.add((level, label, node["content"]))
            gold_level2overall[level].add((label, node["content"]))
            if label == "Heading":
                gold_level2heading[level].add(node["content"])
            else:
                gold_level2text[level].add(node["content"])

        heading_max_depth = max(
            [
                1,
                max([1, *pred_level2heading.keys()]),
                max([1, *gold_level2heading.keys()]),
            ]
        )

        for i in range(1, heading_max_depth + 1):
            tp = len(pred_level2heading[i] & gold_level2heading[i])
            fp = len(pred_level2heading[i] - gold_level2heading[i])
            fn = len(gold_level2heading[i] - pred_level2heading[i])
            heading_result["tp"] += tp
            heading_result["fp"] += fp
            heading_result["fn"] += fn

        text_max_depth = max(
            [
                1,
                max([1, *pred_level2text.keys()]),
                max([1, *gold_level2text.keys()]),
            ]
        )

        for i in range(1, text_max_depth + 1):
            tp = len(pred_level2text[i] & gold_level2text[i])
            fp = len(pred_level2text[i] - gold_level2text[i])
            fn = len(gold_level2text[i] - pred_level2text[i])
            text_result["tp"] += tp
            text_result["fp"] += fp
            text_result["fn"] += fn


    final_metrics["heading"] = calc_p_r_f1_from_tp_fp_fn(
        heading_result["tp"], heading_result["fp"], heading_result["fn"]
    )
    final_metrics["text"] = calc_p_r_f1_from_tp_fp_fn(
        text_result["tp"], text_result["fp"], text_result["fn"]
    )
    overall_result["tp"] = len(pred_nodes & gold_nodes)
    overall_result["fp"] = len(pred_nodes - gold_nodes)
    overall_result["fn"] = len(gold_nodes - pred_nodes)
    final_metrics["overall"] = calc_p_r_f1_from_tp_fp_fn(
        overall_result["tp"], overall_result["fp"], overall_result["fn"]
    )

    return final_metrics