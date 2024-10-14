import json
import os
from typing import List

from pathlib import Path
import fire

from seg2act.utils.node import Node


def document_to_raw_segments(json_obj: dict) -> List[str]:
    segment_list = []
    for page_lines in json_obj['lines']:
        segment_list.extend(page_lines)
    return ["ROOT", "[TITLE]"] + [segment['content'] for segment in segment_list]

def document_to_gold_answers(json_obj: dict) -> List[dict]:
    segment_list = []
    for page_lines in json_obj["lines"]:
        segment_list.extend(page_lines)
    
    # HierDoc only provides nodes excluding Title, so add a virtual one
    node_list : List[Node] = [Node(content='[TITLE]', label='Heading', depth=1)]
    last_heading_depth = 1
    for segment in segment_list:
        if segment['is_title']:
            if segment['relation'] == 'contain':
                depth = node_list[segment['parent']].depth + 1
                label = 'Heading'
            elif segment['relation'] == 'equal':
                depth = node_list[segment['parent']].depth
                label = 'Heading'
            elif segment['relation'] == 'sibling':
                depth = node_list[segment['parent']].depth
                label = 'Concat'
            node_list.append(Node(content=segment['content'], label=label, depth=depth))
            last_heading_depth = depth
        else:
            if segment["line_id"] > 1 and not segment_list[segment["line_id"] - 2]['is_title']:
                node_list.append(Node(
                    content=segment['content'], label='Concat', depth=last_heading_depth + 1))
            else:
                node_list.append(Node(
                    content=segment['content'], label='Text', depth=last_heading_depth + 1))
    node_list.insert(0, Node(content='ROOT', label='Root', depth=0))

    answer_list = []
    for node in node_list:
        if node.label != 'Concat':
            answer_list.append(node.to_dict())
        else:
            assert answer_list[-1]['label'] in ['Heading', 'Text']
            answer_list[-1]['content'] += ' ' + node.content

    return answer_list

def preprocess(input_file: str, output_file: str):
    results = []
    with open(input_file, "r") as reader:
        json_objs = json.load(reader)
        for json_obj in json_objs:
            segments = document_to_raw_segments(json_obj)
            answers = document_to_gold_answers(json_obj)
            results.append({'segments': segments, 'answers': answers})
    print(f"total data: {len(results)}")
    os.makedirs(Path(output_file).parent, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    fire.Fire(preprocess)
