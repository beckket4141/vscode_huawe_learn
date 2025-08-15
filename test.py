n = int(input())
nums = list(map(int, input().split()))
total = int(input())

# 如果总和是奇数，直接返回0
target = total
    # 初始化dp数组，dp[j]表示和为j的方案数
dp = [0] * (target + 1)
dp[0] = 1  # 空子集的和为0，方案数为1
    
for num in nums:
    # 从target逆向遍历到num，避免重复使用当前元素
    for j in range(target, num - 1, -1):
        dp[j] += dp[j - num]
    
print(dp[target])