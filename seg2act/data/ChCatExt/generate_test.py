import json
import os
from typing import List

from pathlib import Path
import fire
import jsonlines


def document_to_raw_segments(json_obj: dict) -> List[str]:
    segment_list = []
    def travel(obj: dict):
        for i in range(len(obj["content"])):
            obj["content"][i] = obj["content"][i].replace('\n', '')
            
        segment_list.extend(obj["content"])
        if len(obj["children"]) > 0:
            for child in obj["children"]:
                travel(child)
    travel(json_obj)
    return segment_list

def document_to_gold_answers(json_obj: dict) -> List[dict]:
    answer_list = []
    def travel(obj: dict, depth: int):
        for i in range(len(obj["content"])):
            obj["content"][i] = obj["content"][i].replace('\n', '')
        answer_list.append({
            "content": "".join(obj["content"]),
            "label": obj["label"],
            "depth": depth,
        })
        if len(obj["children"]) > 0:
            for child in obj["children"]:
                travel(child, depth + 1)
    travel(json_obj, 0)
    return answer_list

def preprocess(input_file: str, output_file: str):
    results = []
    with jsonlines.open(input_file, "r") as reader:
        for json_obj in reader:
            segments = document_to_raw_segments(json_obj)
            answers = document_to_gold_answers(json_obj)
            results.append({'segments': segments, 'answers': answers})
    print(f"total data: {len(results)}")
    os.makedirs(Path(output_file).parent, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    fire.Fire(preprocess)
