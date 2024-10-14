import json
import os
from typing import Any, List, Optional

from pathlib import Path
import fire
import jsonlines

from seg2act.utils.utils import cut
from seg2act.utils.node import Node
from seg2act.data.ChCatExt.tracer_format import travel


def generate_stack_prompt(last_node: Node, last_content: str):
    node_stack = []
    while last_node.parent is not None:
        last_node = last_node.parent
        if last_node.label == "Heading":
            node_stack.append(cut("".join(last_node.content)))
    stack_prompt = ""
    for t in node_stack[::-1]:
        stack_prompt += f"{t}\n"
    stack_prompt += f"{cut(last_content)}\n"
    return stack_prompt

def document_to_datapoints(json_obj: dict):
    datapoint_list = []
    node_list: List[Node] = []

    def build_tree(node: Node):
        if len(node.content) > 1:
            for i in range(1, len(node.content)):
                datapoint_list.append({
                    "tree": generate_stack_prompt(node, "".join(node.content[:i])),
                    "input": f"{cut(node.content[i])}\n",
                    "output": "=\n",
                })
        if len(node_list) > 0:
            last_node = node_list[-1]
            last_depth = last_node.depth
            current_depth = node.depth
            if current_depth <= last_depth:
                # reduce
                while current_depth <= last_depth:
                    datapoint_list.append({
                        "tree": generate_stack_prompt(last_node, "".join(last_node.content)),
                        "input": f"{cut(node.content[0])}\n",
                        "output": "-\n",
                    })
                    last_node = last_node.parent
                    last_depth = last_node.depth
            last_content = "".join(last_node.content)
            # sub
            datapoint_list.append({
                "tree": generate_stack_prompt(last_node, last_content) if last_content != "ROOT" else "",
                "input": f"{cut(node.content[0])}\n",
                "output": {"Heading": "+", "Text": "*"}[node.label] + "\n",
            })
        node_list.append(node)
        if len(node.children) > 0:
            for child in node.children:
                build_tree(child)


    root_node = travel(json_obj, 0)
    build_tree(root_node)
    return datapoint_list

def preprocess(input_file: str, output_file: str):
    results = []
    with jsonlines.open(input_file, "r") as reader:
        for json_obj in reader:
            datapoints = document_to_datapoints(json_obj)
            results.extend(datapoints)
    print(f"Total data: {len(results)}")
    os.makedirs(Path(output_file).parent, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    fire.Fire(preprocess)

