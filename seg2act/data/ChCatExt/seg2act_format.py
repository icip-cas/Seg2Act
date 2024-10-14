import json
import os
from typing import Any, List, Optional

from pathlib import Path
import fire
import jsonlines

from seg2act.utils.utils import cut
from seg2act.utils.node import Node


def generate_datapoint(node_stack: List[Node], possible_node_list: List[Node], cutter=cut):
    def generate_stack_prompt(stack : List[Node]):
        stack_prompt = ""
        for i, node in enumerate(stack):
            if i + 1 < len(stack) and stack[i + 1].depth == node.depth:
                continue
            flag = "+" * node.depth if node.label == "Heading" else "*"
            content = node.content
            stack_prompt += f"{flag} {cutter(content)}\n"
        return stack_prompt
    
    def generate_segment_and_action_prompt(possible_node_list: List[Node]):
        segment_prompt = ""
        action_prompt = ""
        for k in range(len(possible_node_list)):
            content = possible_node_list[k].content
            segment_prompt += f"{cutter(content)}\n"
            if possible_node_list[k].label == "Heading":
                action_prompt += "+" * possible_node_list[k].depth + "\n"
            elif possible_node_list[k].label == "Text":
                action_prompt += "*" + "\n"
            elif possible_node_list[k].label == "Concat":
                action_prompt += "=" + "\n"
        return segment_prompt, action_prompt

    stack_prompt = generate_stack_prompt(node_stack)
    segment_prompt, action_prompt = generate_segment_and_action_prompt(possible_node_list)

    return {
        "tree": stack_prompt,
        "input": segment_prompt,
        "output": action_prompt,
    }

def document_to_datapoints(json_obj: dict, max_seq: int = 3):
    datapoint_list = []
    node_list : List[Node] = []

    def travel(obj: dict, depth: int):
        content = obj["content"]
        count = len(content)
        labels = [obj["label"]] + ["Concat"] * (count - 1)
        depths = [depth] * count
        node_list.extend([Node(depth=j[0], label=j[1], content=j[2]) 
                          for j in zip(depths, labels, content)])
        if len(obj["children"]) > 0:
            for child in obj["children"]:
                travel(child, depth + 1)

    def build_logical_tree():
        node_stack : List[Node] = []
        reduce_map = {}
        for i in range(1, len(node_list)):
            datapoint_list.append(generate_datapoint(
                node_stack=node_stack.copy(),
                possible_node_list=node_list[i: i + max_seq],
            ))
            if node_list[i].depth == node_list[i - 1].depth:
                if node_list[i].label in ["Heading", "Text"]:
                    if len(node_stack) > 0:
                        node_stack = node_stack[:-1]
                    node_stack.append(node_list[i])
                else:
                    # concat
                    node_stack[-1].content += node_list[i].content
            elif node_list[i].depth < node_list[i - 1].depth:
                # reduce + sub
                node_stack = node_stack[:reduce_map[node_list[i].depth]]
                node_stack.append(node_list[i])
            else:
                # sub
                node_stack.append(node_list[i])
            reduce_map[node_list[i].depth] = len(node_stack) - 1
    
    travel(json_obj, 0)
    build_logical_tree()
    return datapoint_list

def preprocess(input_file: str, output_file: str, max_seq: int = 3):
    results = []
    with jsonlines.open(input_file, "r") as reader:
        for json_obj in reader:
            datapoints = document_to_datapoints(json_obj, max_seq=max_seq)
            results.extend(datapoints)
    print(f"Total data: {len(results)}")
    os.makedirs(Path(output_file).parent, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    fire.Fire(preprocess)
