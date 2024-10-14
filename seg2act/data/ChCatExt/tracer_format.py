import json
import os
from typing import Any, List, Optional

from pathlib import Path
import fire
import jsonlines

from seg2act.utils.utils import cut
from seg2act.utils.node import Node


def travel(obj: dict, depth: int):

    if len(obj["children"]) == 0:
        node = Node(content=obj["content"], label=obj["label"], depth=depth)
        return node
    else:
        children_nodes = []
        for child in obj["children"]:
            node = travel(child, depth + 1)
            children_nodes.append(node)
        current_node = Node(
            content=obj["content"], label=obj["label"], 
            depth=depth, children=children_nodes, 
        )
        for child_node in current_node.children:
            child_node.parent = current_node
        return current_node

def document_to_datapoints(json_obj: dict):
    datapoint_list = []
    node_list: List[Node] = []

    def build_tree(node: Node):
        if len(node.content) > 1:
            for i in range(1, len(node.content)):
                datapoint_list.append({
                    "tree": f"{cut(''.join(node.content[:i]))}\n",
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
                        "tree": f"{cut(''.join(last_node.content))}\n",
                        "input": f"{cut(node.content[0])}\n",
                        "output": "-\n",
                    })
                    last_node = last_node.parent
                    last_depth = last_node.depth
            last_content = "".join(last_node.content)
            # sub
            datapoint_list.append({
                "tree": f"{cut(last_content)}\n" if last_content != "ROOT" else "",
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
