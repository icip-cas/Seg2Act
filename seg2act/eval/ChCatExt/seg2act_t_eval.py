import json
import os
import time
from typing import List

import fire
import torch
from transformers import GenerationConfig
from tqdm import tqdm

from seg2act.data.ChCatExt.seg2act_t_format import generate_stack_prompt
from seg2act.utils.utils import cut, generate_prompt
from seg2act.utils.node import Node
from seg2act.eval.eval_utils import load_model
from seg2act.eval.ChCatExt.metrics import calc_hierarchical_metrics
from seg2act.eval.ChCatExt.tracer_eval import run_model, decode_output


@torch.no_grad()
def inference_one_document(
    segment_list: List[str], 
    model, 
    tokenizer, 
    generation_config, 
    max_total_len=512,
):
    if len(segment_list) > 0 and segment_list[0] == 'ROOT':
        segment_list = segment_list[1:]
    
    root_node = Node(depth=0, label="Root", content=["ROOT"])
    first_node = Node(depth=1, label="Heading", content=[segment_list[0]])
    root_node.children.append(first_node)
    first_node.parent = root_node

    last_node = first_node
    for i in range(1, len(segment_list)):
        current_content = segment_list[i]
        while current_content is not None:
            prompt = generate_prompt({
                "tree": f"{generate_stack_prompt(last_node, ''.join(last_node.content))}", 
                "input": f"{cut(current_content)}\n",
                "output": "",
            })
            output = run_model(prompt, model, tokenizer, generation_config, max_total_len)
            if last_node.label == "Root":
                possible_actions = ["+", "*"]
            elif last_node.label == "Text":
                possible_actions = ["-", "="]
            else:
                possible_actions = ["+", "*", "-", "="]
            answer = decode_output(output, possible_actions)
            if answer == "Concat":
                last_node.content.append(current_content)
                current_content = None
            elif answer == "SubHeading":
                new_node = Node(
                    depth=last_node.depth + 1, 
                    label="Heading", 
                    content=[current_content],
                    parent=last_node,
                )
                last_node.children.append(new_node)
                last_node = new_node
                current_content = None
            elif answer == "SubText":
                new_node = Node(
                    depth=last_node.depth + 1, 
                    label="Text", 
                    content=[current_content],
                    parent=last_node,
                )
                last_node.children.append(new_node)
                last_node = new_node
                current_content = None
            elif answer == "Reduce":
                last_node = last_node.parent
    
    node_list = []
    def travel(node: Node):
        d = node.to_dict()
        node_list.append(d)
        if len(node.children) > 0:
            for child in node.children:
                travel(child)
    travel(root_node)
    return node_list

@torch.no_grad()
def seg2act_t_eval(
    # model / data params
    data_path: str, exp_dir: str, pred_name: str,
    base_model: str = "", 
    overwrite: bool = False,  
    max_total_len: int = 512,
):
    pred_path = os.path.join(exp_dir, f"{pred_name}.json")
    use_cache_result = True
    if not (os.path.exists(pred_path) and not overwrite):
        use_cache_result = False
        print("Generating the logical structures of the documents...")
        model, tokenizer = load_model(base_model, exp_dir)
        generation_config = GenerationConfig.from_pretrained(
            base_model, max_new_tokens=3,
            return_dict_in_generate=True, output_scores=True, 
        )
        with open(data_path, "r") as f_r:
            with open(pred_path, "w", encoding="utf-8") as f_w:
                reader = json.load(f_r)
                tot_time = 0.0
                for json_obj in tqdm(reader):
                    stime = time.time()
                    pred = inference_one_document(
                        json_obj['segments'], model, tokenizer, 
                        generation_config, max_total_len)
                    utime = time.time() - stime
                    tot_time += utime
                    json_obj['preds'] = pred

                json.dump(reader, f_w, indent=4, ensure_ascii=False)
                doc_time = tot_time / len(reader)
                print(f"Total Time = {tot_time:.3f} s")
                print(f"Average Time Per Document = {doc_time:.3f} s")

    if use_cache_result:
        print(f"Using the results of cached logical structures."
              f"(If you want to re-generate, please set overwrite=True)")

    with open(pred_path, 'r', encoding="utf-8") as f_r:
        x = json.load(f_r)
        preds = [i['preds'] for i in x]
        golds = [i['answers'] for i in x]
        metric = calc_hierarchical_metrics(preds, golds)
        for x1 in ["heading", "text", "overall"]:
            print(f"{x1.capitalize()}: ")
            for x2 in ["p", "r", "f1"]:
                print(f"\t{x2.capitalize()}: {metric[x1][x2] * 100:.3f}\t")
        doc_acc = sum([1 if pred == gold else 0 for pred, gold in zip(preds, golds)]) / len(x)
        print(f"Document Accurancy = {doc_acc * 100.0:.3f}")


if __name__ == "__main__":
    fire.Fire(seg2act_t_eval)