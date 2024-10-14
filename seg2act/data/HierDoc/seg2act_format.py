import json
import os
from typing import Any, List, Optional

from pathlib import Path
import fire
import jsonlines

from seg2act.utils.utils import english_cut
from seg2act.utils.node import Node
from seg2act.data.ChCatExt.seg2act_format import generate_datapoint


def document_to_datapoints(json_obj: dict, max_seq: int = 3):
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

    datapoint_list = []
    def build_logical_tree():
        node_stack : List[Node] = []
        reduce_map = {}
        for i in range(1, len(node_list)):
            datapoint_list.append(generate_datapoint(
                node_stack=node_stack.copy(),
                possible_node_list=node_list[i: i + max_seq],
                cutter=english_cut,
            ))
            if node_list[i].depth == node_list[i - 1].depth:
                if node_list[i].label in ["Heading", "Text"]:
                    if len(node_stack) > 0:
                        node_stack = node_stack[:-1]
                    node_stack.append(node_list[i])
                else:
                    # concat
                    node_stack[-1].content += " " + node_list[i].content
            elif node_list[i].depth < node_list[i - 1].depth:
                # reduce + sub
                node_stack = node_stack[:reduce_map[node_list[i].depth]]
                node_stack.append(node_list[i])
            else:
                # sub
                node_stack.append(node_list[i])
            reduce_map[node_list[i].depth] = len(node_stack) - 1
    
    build_logical_tree()
    return datapoint_list

def preprocess(input_file: str, output_file: str, max_seq: int = 3):
    results = []
    with open(input_file, "r") as reader:
        json_objs = json.load(reader)
        for json_obj in json_objs:
            try:
                datapoints = document_to_datapoints(json_obj, max_seq=max_seq)
                results.extend(datapoints)
            except:
                continue
    print(f"Total data: {len(results)}")
    os.makedirs(Path(output_file).parent, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    fire.Fire(preprocess)
