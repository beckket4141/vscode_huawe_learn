# 滑动窗口算法伪码框架
def slidingWindow(s: str):
    # 用合适的数据结构记录窗口中的数据，根据具体场景变通
    # 比如说，我想记录窗口中元素出现的次数，就用 map
    # 如果我想记录窗口中的元素和，就可以只用一个 int
    window = ...

    left, right = 0, 0
    while right < len(s):
        # c 是将移入窗口的字符
        c = s[right]
        window.add(c)
        # 增大窗口
        right += 1
        # 进行窗口内数据的一系列更新
        ...

        # *** debug 输出的位置 ***
        # 注意在最终的解法代码中不要 print
        # 因为 IO 操作很耗时，可能导致超时
        # print(f"window: [{left}, {right})")
        # ***********************

        # 判断左侧窗口是否要收缩
        while left < right and window needs shrink:
            # d 是将移出窗口的字符
            d = s[left]
            window.remove(d)
            # 缩小窗口
            left += 1
            # 进行窗口内数据的一系列更新
            ...


            #eg：


class Solution:
        # 判断 s 中是否存在 t 的排列
        def checkInclusion(self, t: str, s: str) -> bool:
            need = {}
            window = {}
            for c in t:
                need[c] = need.get(c, 0) + 1

            left = 0
            right = 0
            valid = 0
            while right < len(s):
                c = s[right]
                right += 1
                # 进行窗口内数据的一系列更新
                if c in need:
                    window[c] = window.get(c, 0) + 1
                    if window[c] == need[c]:
                        valid += 1

                # 判断左侧窗口是否要收缩
                while right - left >= len(t):
                    # 在这里判断是否找到了合法的子串
                    if valid == len(need):
                        return True
                    d = s[left]
                    left += 1
                    # 进行窗口内数据的一系列更新
                    if d in need:
                        if window[d] == need[d]:
                            valid -= 1
                        window[d] -= 1

            # 未找到符合条件的子串
            return False