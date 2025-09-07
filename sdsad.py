# Python 版本
import sys
sys.setrecursionlimit(10000)

# 读入并分词
tokens = sys.stdin.read().strip().split()
pos = 0

class Node:
    def calc(self): pass

class TokenNode(Node):
    def __init__(self, tok): self.tok = tok
    def calc(self):
        return {self.tok: 1}

class SequenceNode(Node):
    def __init__(self): self.children = []
    def calc(self):
        res = {}
        for c in self.children:
            for k,v in c.calc().items():
                res[k] = res.get(k, 0) + v
        return res

class BranchNode(Node):
    def __init__(self, required): self.required = required; self.options = []
    def calc(self):
        maps = [opt.calc() for opt in self.options]
        if not self.required:
            maps.append({})
        keys = set(k for m in maps for k in m)
        res = {}
        for k in keys:
            mn = min(m.get(k,0) for m in maps)
            if mn>0: res[k] = mn
        return res

def parse_branch():
    global pos
    req = tokens[pos] == '{'
    pos += 1
    bn = BranchNode(req)
    while tokens[pos] != ('}' if req else ']'):
        seq = SequenceNode()
        while tokens[pos] not in ['|', '}', ']']:
            if tokens[pos] in ['{','[']:
                seq.children.append(parse_branch())
            else:
                seq.children.append(TokenNode(tokens[pos]))
                pos += 1
        bn.options.append(seq)
        if tokens[pos] == '|': pos += 1
    pos += 1
    return bn

def parse_node():
    global pos
    seq = SequenceNode()
    while pos < len(tokens) and tokens[pos] not in ['}',''] :
        if tokens[pos] in ['{','[']:
            seq.children.append(parse_branch())
        elif tokens[pos] in ['}','|',']']:
            break
        else:
            seq.children.append(TokenNode(tokens[pos]))
            pos += 1
    return seq

root = parse_node()
mp = root.calc()
ans = sorted((k,v) for k,v in mp.items() if v>0)
print(' '.join(k for k,_ in ans))
print(' '.join(str(v) for _,v in ans))
