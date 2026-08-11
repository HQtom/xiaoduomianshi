"""
幻觉检测结果验证脚本

功能：
    对照 ground_truth_reference.json（人工标注的真实结果）与
    output.json（eval_script.py 调用DeepSeek生成的检测结果），
    逐条比较 is_hallucination 判断是否一致，统计检出率相关指标
    （准确率/精确率/召回率/F1），并标记出判断错误或类型不一致的记录，
    最终将详细对照结果与汇总统计写入 output_eval.json。

注意：
    output.json 中的 hallucination_type 采用五类新分类体系
    （虚构事实/参数和事实错误/政策规则错误/安全风险/未采纳文档），
    与 ground_truth_reference.json 中的类型标签（如 政策编造/参数编造/
    能力越界 等）命名体系不同，因此 type_match 字段仅作参考信息，
    核心检出率指标以 is_hallucination 的一致性为准。

用法：
    python eval_verify.py
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "ground_truth_reference.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "output.json")
OUTPUT_EVAL_PATH = os.path.join(BASE_DIR, "output_eval.json")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_by_id(items: list) -> dict:
    return {item["id"]: item for item in items}


def main() -> None:
    ground_truth = load_json(GROUND_TRUTH_PATH)
    predictions = load_json(OUTPUT_PATH)

    gt_map = index_by_id(ground_truth)
    pred_map = index_by_id(predictions)

    all_ids = sorted(set(gt_map.keys()) | set(pred_map.keys()))

    details = []
    tp = fp = fn = tn = 0
    missing_in_pred = []
    missing_in_gt = []
    type_mismatch_count = 0

    for item_id in all_ids:
        gt = gt_map.get(item_id)
        pred = pred_map.get(item_id)

        if gt is None:
            missing_in_gt.append(item_id)
            continue
        if pred is None:
            missing_in_pred.append(item_id)
            continue

        gt_is_hallu = gt.get("is_hallucination")
        pred_is_hallu = pred.get("is_hallucination")
        gt_type = gt.get("hallucination_type")
        pred_type = pred.get("hallucination_type")

        is_hallu_correct = gt_is_hallu == pred_is_hallu

        if gt_is_hallu is True and pred_is_hallu is True:
            tp += 1
        elif gt_is_hallu is False and pred_is_hallu is True:
            fp += 1
        elif gt_is_hallu is True and pred_is_hallu is False:
            fn += 1
        elif gt_is_hallu is False and pred_is_hallu is False:
            tn += 1

        type_match = None
        if gt_is_hallu and pred_is_hallu:
            type_match = gt_type == pred_type
            if not type_match:
                type_mismatch_count += 1

        details.append(
            {
                "id": item_id,
                "gt_is_hallucination": gt_is_hallu,
                "pred_is_hallucination": pred_is_hallu,
                "is_hallucination_correct": is_hallu_correct,
                "gt_hallucination_type": gt_type,
                "pred_hallucination_type": pred_type,
                "type_match": type_match,
                "gt_detail": gt.get("detail"),
                "pred_reason": pred.get("reason"),
            }
        )

    total_compared = len(details)
    correct_count = sum(1 for d in details if d["is_hallucination_correct"])
    accuracy = correct_count / total_compared if total_compared else None

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    anomalies = [
        d["id"] for d in details if not d["is_hallucination_correct"]
    ] + missing_in_pred + missing_in_gt

    summary = {
        "total_ground_truth": len(ground_truth),
        "total_predictions": len(predictions),
        "total_compared": total_compared,
        "is_hallucination_accuracy": accuracy,
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        },
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "hallucination_type_mismatch_count": type_mismatch_count,
        "missing_in_output": missing_in_pred,
        "missing_in_ground_truth": missing_in_gt,
        "anomaly_ids": anomalies,
    }

    result = {
        "summary": summary,
        "details": details,
    }

    with open(OUTPUT_EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("== 检出率验证汇总 ==")
    print(f"比较记录数：{total_compared}")
    print(f"is_hallucination 准确率：{accuracy:.2%}" if accuracy is not None else "is_hallucination 准确率：N/A")
    print(f"Precision：{precision:.2%}" if precision is not None else "Precision：N/A")
    print(f"Recall：{recall:.2%}" if recall is not None else "Recall：N/A")
    print(f"F1：{f1:.2%}" if f1 is not None else "F1：N/A")
    print(f"混淆矩阵：TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"幻觉类型不一致数（新旧分类体系不同，仅供参考）：{type_mismatch_count}")
    if anomalies:
        print(f"异常/判断错误的id：{anomalies}")
    else:
        print("未发现is_hallucination判断异常。")
    print(f"详细结果已保存至 {OUTPUT_EVAL_PATH}")


if __name__ == "__main__":
    main()
