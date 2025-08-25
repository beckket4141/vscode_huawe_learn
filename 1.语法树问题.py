'''
解决的问题: 语法树问题（命令关键字统计）
题目描述:
    给定一个命令格式字符串，包含普通关键字和分支结构（必
    必选关键字）和可选关键字），要求计算出所有必选关键字的最小出现次数。
    例如：d r { k | n k } [ k | n k ]
    解析后可知，关键字d和r是必选的，k和n是分支内的关键字。
    由于{ k | n k }是必选分支，无论选哪个选项，k至少出现1次，因此k也是必选关键字。
    而[n k | n k]是可选分支，可以不选，因此k          是必选关键字。
    因此，必选关键字的最小出现次数为2。
    输入：d r { k | n k } [ k | n k ]
    输出： d r k
          1 1 2  
'''





import sys
# 提高递归栈限制（因为命令格式可能有深层嵌套，避免递归调用时栈溢出）
sys.setrecursionlimit(10000)

class Solution:
    """
    命令关键字统计问题的解决方案类
    功能：解析命令格式字符串，计算必选关键字的最小出现次数并输出
    核心思路：通过递归解析生成语法树，再遍历树计算结果
    """
    def __init__(self):
        """初始化方法：读取输入、分词、初始化解析指针"""
        # 读取输入的命令格式字符串，按空格拆分为令牌（tokens）
        # 例如输入"d r { k | n k }"会拆分为["d", "r", "{", "k", "|", "n", "k", "}"]
        self.tokens = sys.stdin.read().strip().split()
        # 解析指针：记录当前正在处理的tokens索引（类似"当前读取位置"）
        self.pos = 0  
    
    # ------------------------------
    # 内部节点类：封装不同结构的"数据"和"计算逻辑"
    # ------------------------------
    class Node:
        """基类（抽象节点）：定义所有节点的统一接口calc"""
        def calc(self):
            """计算当前节点对应的关键字出现次数（子类必须实现）"""
            pass
    
    class TokenNode(Node):
        """普通关键字节点：处理单个关键字（如"d"、"k"）"""
        def __init__(self, tok):
            # 存储关键字内容（如tok="d"）
            self.tok = tok
        
        def calc(self):
            """计算逻辑：单个关键字出现次数固定为1"""
            return {self.tok: 1}  # 返回{关键字: 次数}的字典
    
    class SequenceNode(Node):
        """顺序节点：处理按顺序排列的元素（如"d r { ... }"中的顺序结构）"""
        def __init__(self):
            # 存储子节点列表（子节点可以是TokenNode/BranchNode/SequenceNode）
            self.children = []
        
        def calc(self):
            """计算逻辑：累加所有子节点的关键字出现次数"""
            res = {}  # 存储最终的关键字次数字典
            for child in self.children:
                # 递归调用子节点的calc()，获取子节点的计算结果
                child_res = child.calc()
                # 累加每个关键字的次数（若关键字已存在则相加，否则新增）
                for k, v in child_res.items():
                    res[k] = res.get(k, 0) + v  # res.get(k,0)表示默认值为0
            return res
    
    class BranchNode(Node):
        """分支节点：处理带选项的分支结构（{...}必选分支或[...]可选分支）"""
        def __init__(self, required):
            self.required = required  # True表示{...}必选，False表示[...]可选
            self.options = []  # 存储分支内的选项列表（每个选项是SequenceNode）
        
        def calc(self):
            """计算逻辑：取所有选项中关键字出现次数的最小值（必选关键字的核心逻辑）"""
            # 1. 计算每个选项的关键字次数（每个选项是SequenceNode，递归调用calc()）
            option_res_list = [opt.calc() for opt in self.options]
            
            # 2. 可选分支额外添加"空选项"（表示不选该分支，所有关键字次数为0）
            if not self.required:
                option_res_list.append({})  # 空字典代表次数为0
            
            # 3. 收集所有选项中出现过的关键字（去重）
            all_keys = set()
            for res in option_res_list:
                all_keys.update(res.keys())  # 把每个选项的关键字加入集合（自动去重）
            
            # 4. 计算每个关键字在所有选项中的最小出现次数
            res = {}
            for key in all_keys:
                # 遍历所有选项，获取当前关键字的次数（若选项中没有该关键字，默认次数为0）
                min_count = min(res.get(key, 0) for res in option_res_list)
                # 只有最小次数>0时，该关键字才是必选的（无论选哪个选项都至少出现这么多次）
                if min_count > 0:
                    res[key] = min_count
            return res
    
    # ------------------------------
    # 解析方法：将tokens转换为语法树（递归核心）
    # ------------------------------
    def parse_branch(self):
        """解析分支结构（{...}或[...]），返回BranchNode对象"""
        # 1. 判断分支类型（必选/可选）：当前令牌是'{'则为必选，'['则为可选
        is_required = self.tokens[self.pos] == '{'
        # 移动指针：跳过当前的'{'或'['（处理下一个令牌）
        self.pos += 1
        
        # 2. 创建分支节点（记录是否必选）
        branch_node = self.BranchNode(is_required)
        
        # 3. 循环解析分支内的所有选项（直到遇到闭合符号'}'或']'）
        # 闭合符号规则：必选分支找'}'，可选分支找']'
        while self.tokens[self.pos] != ('}' if is_required else ']'):
            # 每个选项都是"顺序结构"，用SequenceNode存储
            option_seq = self.SequenceNode()
            
            # 解析当前选项内的元素（直到遇到选项分隔符'|'或分支闭合符号）
            while self.tokens[self.pos] not in ['|', '}', ']']:
                current_token = self.tokens[self.pos]
                if current_token in ['{', '[']:
                    # 遇到嵌套的分支结构，递归调用parse_branch()解析
                    # 递归后返回子BranchNode，加入当前选项的子节点列表
                    option_seq.children.append(self.parse_branch())
                else:
                    # 遇到普通关键字，创建TokenNode加入当前选项的子节点列表
                    option_seq.children.append(self.TokenNode(current_token))
                    # 移动指针：处理下一个令牌
                    self.pos += 1
            
            # 将解析好的当前选项加入分支节点的选项列表
            branch_node.options.append(option_seq)
            
            # 若当前令牌是选项分隔符'|'，移动指针跳过（准备解析下一个选项）
            if self.tokens[self.pos] == '|':
                self.pos += 1
        
        # 4. 解析完所有选项后，移动指针跳过闭合符号'}'或']'
        self.pos += 1
        
        # 返回解析好的分支节点
        return branch_node
    
    def parse_node(self):
        """解析顺序结构（默认的元素排列方式），返回SequenceNode对象"""
        # 创建顺序节点，存储按顺序排列的元素
        seq_node = self.SequenceNode()
        
        # 循环解析令牌，直到指针越界或遇到分支闭合符号（退出当前顺序结构）
        while self.pos < len(self.tokens) and self.tokens[self.pos] not in ['}']:
            current_token = self.tokens[self.pos]
            if current_token in ['{', '[']:
                # 遇到分支结构，调用parse_branch()解析，返回的BranchNode加入子节点
                seq_node.children.append(self.parse_branch())
            elif current_token in ['}', '|', ']']:
                # 遇到分支内的特殊符号（闭合符或分隔符），退出当前顺序解析
                break
            else:
                # 遇到普通关键字，创建TokenNode加入子节点列表
                seq_node.children.append(self.TokenNode(current_token))
                # 移动指针：处理下一个令牌
                self.pos += 1
        
        # 返回解析好的顺序节点
        return seq_node
    
    # ------------------------------
    # 主方法：统筹解析、计算、输出
    # ------------------------------
    def solve(self):
        """执行整个流程：解析命令格式→计算结果→输出"""
        # 1. 从根节点开始解析（根节点是最外层的顺序结构）
        root_node = self.parse_node()
        
        # 2. 递归计算根节点的关键字次数（触发整个语法树的calc()调用）
        keyword_counts = root_node.calc()
        
        # 3. 处理结果：筛选出次数>0的必选关键字，按字母顺序排序
        # 转换为列表并排序（sorted默认按关键字字母顺序）
        sorted_results = sorted((k, v) for k, v in keyword_counts.items() if v > 0)
        
        # 4. 输出结果：第一行关键字，第二行对应次数
        # 提取关键字并拼接为字符串
        print(' '.join(k for k, _ in sorted_results))
        # 提取次数并转换为字符串拼接
        print(' '.join(str(v) for _, v in sorted_results))

# 执行代码：创建Solution实例并调用solve()方法
if __name__ == "__main__":
    solution = Solution()
    solution.solve()



'''
1. root 是语法树的根节点:
类型是 SequenceNode（顺序节点），本质是一个 “树形结构的入口”。
root = SequenceNode()
root.children = [
    TokenNode(tok="d"),  # 第一个子节点：普通关键字"d"
    TokenNode(tok="r"),  # 第二个子节点：普通关键字"r"
    BranchNode(required=True)  # 第三个子节点：必选分支{ k | n k }
    BranchNode(required=False)  # 第四个子节点：可选分支[ k | n k ]
]

2.mp 是 root.calc() 的返回值,是一个字典:
格式为 {关键字: 最小出现次数}，存储了所有 “必选关键字” 及其最少出现次数。

3. root.calc()用递归的方式计算了语法树中每个节点的次数，并返回一个字典。
顺序:
- 步骤 1：root（SequenceNode）调用 calc():
    SequenceNode 的 calc() 逻辑是 “累加所有子节点的结果”，
    因此会遍历 root.children，逐个调用子节点的 calc()：

- 步骤 2：递归调用子节点的 calc()        
    第一个子节点 TokenNode("d")：
        调用 calc() 直接返回 {"d":1}（无递归，基础 case）。
    第二个子节点 TokenNode("r")：
        调用 calc() 直接返回 {"r":1}（无递归）。
    第三个子节点 BranchNode（必选分支 {k | n k}）：
        调用 BranchNode 的 calc()，这会触发更深层的递归：
        maps = [opt.calc() for opt in self.options]  # 遍历分支内的选项

- 步骤 3：汇总结果:
    {"d":1} + {"r":1} + {"k":1} → {"d":1, "r":1, "k":1}，即 mp 的值。
        TokenNode.calc()              BranchNode.calc()  
    存入mp
'''