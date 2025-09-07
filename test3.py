import sys
sys.setrecursionlimit(10**7)

# 父类
class Node:
    def calc(self): #定义抽象方法
        pass
#子类
class TokenNode(Node):
    def __init__(self, tok):
        self.tok = tok
    def calc(self): #定义该子类的父类抽象方法的具体实现
        return {self.tok: 1}

class SequenceNode(Node):
    def __init__(self):
        self.children = []
    def calc(self):
        res = {}
        for c in self.children:
            for k, v in c.calc().items():
                res[k] = res.get(k, 0) + v
        return res

class BranchNode(Node):
    def __init__(self, required):
        self.required = required
        self.options = []
    def calc(self):
        maps = [opt.calc() for opt in self.options]
        if not self.required: # 当情况为[]的时候
            maps.append({})
        keys = set(k for m in maps for k in m) # 处理{}的时候
        res = {}
        for k in keys:
            mn = min(m.get(k, 0) for m in maps) # 对于{|}的时候会取到两边都存在的健的值,对于[|]会由于加入了空字典,最终总是会为0
            if mn > 0:
                res[k] = mn
        return res            

def parse_branch():
    global pos
    req = tokens[pos] == '{'
    pos += 1
    end_char = '}' if req else ']'
    bn = BranchNode(req)
    while tokens[pos] != end_char:
        seq = SequenceNode()
        while tokens[pos] != '|' and tokens[pos] != end_char:
            if tokens[pos] in ['{', '[']:
                seq.children.append(parse_branch())
            else:
                seq.children.append(TokenNode(tokens[pos]))
                pos += 1
        bn.options.append(seq)
        if tokens[pos] == '|':
            pos += 1
    pos += 1
    return bn

def parse_node(): #循环解析所有同级子节点---只显性调用一次
    global pos
    seq = SequenceNode()
    while pos < len(tokens) and tokens[pos] not in ['}', ']' , '', '|']:# not in ['}', '', ']']根本不会触发
        if tokens[pos] in ['{', '[']:
            seq.children.append(parse_branch())
        else:
            seq.children.append(TokenNode(tokens[pos]))
            pos += 1
    return seq

tokens = sys.stdin.readline().strip().split()
pos = 0

root = parse_node()
mp = root.calc()
ans = sorted((k,v) for k,v in mp.items() if v>0)
print(' '.join(k for k,_ in ans))
print(' '.join(str(v) for _,v in ans))