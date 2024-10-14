import json
from apted import APTED, Config
from typing import List


def calc_heading_detection(x):

    p_list = []
    r_list = []
    for y in x:
        gold_head_line = pred_head_line = match_head_line = 0
        answer_idx = pred_idx = 1
        for segment_idx, segment in enumerate(y['segments'][2:]): # skip two virtual node: "ROOT" and "[TITLE]"
            while segment not in y['answers'][answer_idx]['content']:
                answer_idx += 1
            while pred_idx < len(y['preds']) and segment not in y['preds'][pred_idx]['content']:
                pred_idx += 1
            
            y['answers'][answer_idx]['content'] = y['answers'][answer_idx]['content'].replace(segment, '', 1)
            if y['answers'][answer_idx]['label'] == 'Heading':
                gold_head_line += 1
            
            if pred_idx < len(y['preds']):
                y['preds'][pred_idx]['content'] = y['preds'][pred_idx]['content'].replace(segment, '', 1)
                if y['preds'][pred_idx]['label'] == 'Heading':
                    pred_head_line += 1
            
                if y['answers'][answer_idx]['label'] == 'Heading' and y['preds'][pred_idx]['label'] == 'Heading':
                    match_head_line += 1
        
        try:
            p = match_head_line / pred_head_line
        except:
            p = 0
        try:
            r = match_head_line / gold_head_line
        except:
            r = 0
        p_list.append(p)
        r_list.append(r)
        
    p = sum(p_list) / len(x)
    r = sum(r_list) / len(x)
    f1 = 2 * p * r / (p + r)
    return p, r, f1


class Node:
    def __init__(self, content, d='', t=''):
        self.name = content
        self.depth = d
        self.type = t
        self.children : List[Node] = []

def convert_preds_to_tree(pred_list):
    depth2node = dict()
    all_node = []
    for pred in pred_list[1:]:
        node = Node(pred['content'], pred['depth'])
        if pred['label'] != 'Text':
            depth2node[node.depth] = node
            if node.depth > 1:
                depth2node[node.depth - 1].children.append(node)
            all_node.append(node)
    return depth2node[1], len(all_node)

class NewConfig(Config):
    def rename(self, node1, node2):
        """node1:pred, node2:label"""
        if node1.name == node2.name and node1.type == node2.type:
            return 0
        else:
            return 1

def calc_tree_edit_distance_similarity(x):
    teds_total = 0
    for doc in x:
        pred_tree, pred_len = convert_preds_to_tree(doc['preds'])
        gold_tree, gold_len = convert_preds_to_tree(doc['answers'])
        distance = APTED(pred_tree, gold_tree, NewConfig()).compute_edit_distance()
        teds = 1.0 - (float(distance) / max([pred_len, gold_len]))
        teds_total += teds
    
    return (teds_total / len(x)) * 100.0