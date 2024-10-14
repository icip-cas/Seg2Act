from typing import Any, List, Optional


class Node:
    def __init__(
        self,
        depth: Optional[int] = None,
        label: Optional[str] = None,
        content: Optional[List[str]] = list(),
        children: Optional[List] = None,
        parent: Optional[Any] = None,
    ):
        self.depth = depth
        self.content = content
        self.label = label
        self.children = children if children else []
        self.parent = parent
    
    def to_dict(self):
        return {
            'content': "".join(self.content) if type(self.content) == list else self.content,
            'label': self.label,
            'depth': self.depth
        }