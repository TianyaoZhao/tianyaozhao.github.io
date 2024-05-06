// https://www.luogu.com.cn/problem/P1025
// 数n分成k份相加，求有几种方案
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
void dfs(int x, int sum, int cnt){
    if(cnt == k){        
        if(sum == n) res ++;
        return;
    }

    for(int i = 1; i < n; i ++){
        if(sum + i > n || cnt + 1 > k) continue; // 左剪枝
    }
}
void solve(){
    cin >> n >> k;
    dfs(0, 0, 0); // dfs(节点编号,当前sum,当前数目cnt) (进入节点之前)
}
int main(){
    io_speed
    solve();
    return 0;
}