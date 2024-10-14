import json
import os
import time
from collections import Counter
from typing import List, Optional

import math
import fire
import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

from seg2act.data.ChCatExt.seg2act_format import generate_datapoint
from seg2act.utils.utils import generate_prompt
from seg2act.utils.node import Node
from seg2act.eval.eval_utils import load_model
from seg2act.eval.ChCatExt.metrics import calc_hierarchical_metrics


class Seg2ActLogitsProcessor(LogitsProcessor):
    def __init__(self, force_start=False) -> None:
        self.dict1 = {
            5: [31340, 31251] # \n -> [+, *]
        }
        self.dict2 = {
            9501: [5, 9501, 31340],  # ++ -> [\n, ++, +]
            31340: [5],  # + -> [\n]
            31191: [5],  # = -> [\n]
            31251: [5],  # * -> [\n]
            5: [9501, 31340, 31191, 31251, 2] # \n -> [++, +, =, *, </s>]
        }
        if force_start:
            self.force_dict = self.dict1
        else:
            self.force_dict = self.dict2
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        assert len(input_ids) == 1, "Require beam_size = 1, batchsize = 1"
        last_token = int(input_ids[0][-1])
        if last_token in self.force_dict:
            mask[0][self.force_dict[last_token]] = 0
        else:
            # input context length > 512 (cannot match "### Output:\n", which is truncated)
            # directly output <pad> or </s> to stop the prediction
            mask[0][[0, 2]] = 0
        self.force_dict = self.dict2
        return scores + mask

@torch.no_grad()
def run_model(prompt, model, tokenizer, generation_config, logits_processors, max_total_len=512):
    batch_inputs = tokenizer(
        [prompt], truncation=True, max_length=max_total_len,
        padding=True, return_tensors="pt").to("cuda")
    raw_outputs = model.generate(
        **batch_inputs,
        generation_config=generation_config,
        logits_processor=logits_processors,
    )
    outputs = tokenizer.batch_decode(raw_outputs, skip_special_tokens=True)
    return outputs[0]

def decode_output(text: str):
    result = text.split("### Output:\n")
    output_result = []
    # filtered if prompt is not endswith "### Output:\n" (truncated)
    if len(result) == 2:
        output_result = result[1].split("\n")[:-1]

    answer = []
    for out in output_result:
        if out == "=":
            answer.append(f"Concat")
        elif out == "*":
            answer.append(f"Text")
        elif len(out) > 0 and out[0] == "+":
            answer.append(f"Heading,{Counter(out)['+']}")
    return answer

@torch.no_grad()
def inference_one_document(
    segment_list: List[str], max_seq, model, tokenizer, 
    generation_config, max_total_len=512, stride=None):

    if len(segment_list) > 0 and segment_list[0] == 'ROOT':
        segment_list = segment_list[1:]
        
    node_list : List[Node] = [Node(content='ROOT', label='Root', depth=0)]
    node_stack : List[Node] = []   
    reduce_map = {}
    for i in range(0, len(segment_list), stride if stride else max_seq):
        text_list = segment_list[i: i + max_seq]
        prompt = generate_prompt({**generate_datapoint(
            node_stack=node_stack,
            possible_node_list=[Node(content=t) for t in text_list],
        ), "output": ""})
        force_start = i == 0
        logits_processors = LogitsProcessorList([Seg2ActLogitsProcessor(force_start)])
        output = run_model(prompt, model, tokenizer, 
                           generation_config, logits_processors, max_total_len)
        answer = decode_output(output)

        # fail to match w_I, we skip these segments
        # if len(answer) != min(max_seq, len(segment_list) - i):continue

        # build node_list and node_stack
        for a, t in zip(answer[:stride], text_list[:stride]):
            # build node_list
            if a.startswith("Heading") or a.startswith("Text"):
                if a.startswith("Heading"):
                    depth = int(a.split(",")[1])
                    if node_list[-1].label in ['Heading', 'Root']:
                        max_depth = node_list[-1].depth + 1
                    else:
                        max_depth = node_list[-1].depth
                    # limit the level of sub-head
                    depth = min(depth, max_depth)  
                    node_list.append(Node(content=t, label='Heading', depth=depth))
                else:
                    depth = 0
                    if len(node_list) > 0:
                        if node_list[-1].label in ["Heading", "Root"]:
                            depth = node_list[-1].depth + 1
                        elif node_list[-1].label == "Text":
                            depth = node_list[-1].depth
                    node_list.append(Node(content=t, label='Text', depth=depth))
                
                # build node_stack
                if len(node_list) > 1 and node_list[-1].depth < node_list[-2].depth:
                    node_stack = node_stack[:reduce_map[node_list[-1].depth]]
                node_stack.append(Node(content=t, label=node_list[-1].label, depth=node_list[-1].depth))
                reduce_map[node_list[-1].depth] = len(node_stack) - 1

            elif a.startswith("Concat") and len(node_list) > 1:
                node_list[-1].content += t
                node_stack[-1].content += t
                
    return [node.to_dict() for node in node_list]

@torch.no_grad()
def seg2act_eval(
    # model / data params
    data_path: str, exp_dir: str, pred_name: str,
    base_model: str = "", 
    overwrite: bool = False, 
    max_total_len: int = 512,
    max_output_len: int = 12,
    max_seq: int = 3,
    stride: Optional[int] = None,
):
    pred_path = os.path.join(exp_dir, f"{pred_name}.json")
    use_cache_result = True
    if not (os.path.exists(pred_path) and not overwrite):
        use_cache_result = False
        print("Generating the logical structures of the documents...")
        model, tokenizer = load_model(base_model, exp_dir)
        generation_config = GenerationConfig.from_pretrained(
            base_model, max_new_tokens=max_output_len,
        )
        with open(data_path, "r") as f_r:
            with open(pred_path, "w", encoding="utf-8") as f_w:
                reader = json.load(f_r)
                tot_time = 0.0
                for json_obj in tqdm(reader):
                    stime = time.time()
                    pred = inference_one_document(
                        json_obj['segments'], max_seq, model, tokenizer, 
                        generation_config, max_total_len, stride)
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
    fire.Fire(seg2act_eval)