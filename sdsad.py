class Solution:
    def mindiv(self, target_list, rank_group):
        self.all_menthod(target_list, rank_group)
        return self.allmethod

    def all_menthod(self, target_list, rank_group):
        self.allmethod = []
        self.curmethod = []
        n = len(target_list)
        rest = rank_group

        def traverse(st, rest):
            rest -= 1
            if rest == 0:
                self.curmethod.append(target_list[st:])
                self.allmethod.append(self.curmethod[:])
                self.curmethod.pop()
                return

            for i in range(st+1, n-rest+1):
                self.curmethod.append(target_list[st:i])
                traverse(i, rest)
                self.curmethod.pop()

        traverse(0, rest)
    

        
a = [1,2,3,4]
k = 3
sol = Solution()
res = sol.mindiv(a,k)
print(res)