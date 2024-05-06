// https://www.luogu.com.cn/problem/P1025
<<<<<<< HEAD
// 数n分成k份相加，求有几种方案
=======
// 将整数n拆分成k份的和，求一共有多少种方案
>>>>>>> e221473eb78db730f77ae49922bd65c0b168e7f7
# include <iostream>
# include <cstring>
# include <algorithm>
# include <string>
# include <cmath>
# include <stack>
# include <queue>
# include <vector>
# define io_speed ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
# define endl '\n'
using namespace std;
typedef pair <int, int> PII;
typedef long long LL;
// const int N = 
int n, k, res;
<<<<<<< HEAD
void dfs(int x, int sum, int cnt){
    if(cnt == k){        
        if(sum == n) res ++;
        return;
    }

    for(int i = 1; i < n; i ++){
        if(sum + i > n || cnt + 1 > k) continue; // 左剪枝
=======
// x代表上一个出现过的数，初值为1，只要让下一个数从x开始循环，便可以实现忽略顺序
void dfs(int x, int sum, int cnt){
    // 递归结束, 找到一个方案
    if(sum == n && cnt == k){
        // cout << "sum" << sum << endl;
        // cout << "cnt" << cnt << endl;
        res ++;
        return;
    }
    for(int i = x; i < n; i ++){
        // 选这个节点，会导致总和超了或者是选择的个数超了，那就不能选, 剪枝
        if(sum + i > n || cnt + 1 > k) break;
        else{
            // cout << "x:" << x << endl;
            // cout << "i:" << i << endl;
            sum += i; 
            cnt ++;
            dfs(i, sum, cnt);
            // 恢复现场
            sum -= i;
            cnt --;
        }
>>>>>>> e221473eb78db730f77ae49922bd65c0b168e7f7
    }
}
void solve(){
    cin >> n >> k;
<<<<<<< HEAD
    dfs(0, 0, 0); // dfs(节点编号,当前sum,当前数目cnt) (进入节点之前)
=======
    dfs(1, 0, 0); // dfs(上一个选择的数, sum, cnt) 到达节点前
    cout << res ;
>>>>>>> e221473eb78db730f77ae49922bd65c0b168e7f7
}
int main(){
    io_speed
    solve();
    return 0;
}