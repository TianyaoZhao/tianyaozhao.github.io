// https://www.luogu.com.cn/problem/P1036
// n个整数选择k个数,是组合数不是排列数,计算和为素数的情况有多少种
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
const int N = 20 + 5;
int n, m, cnt;
int a[N];
vector <int> path;
bool isprim(int x){
    if(x == 1) return false;
    for(int i = 2; i <= x / i; i ++){
        if(x % i == 0) return false;
    }
    return true;
}
void dfs(int u){ // 对下标组合数排列
    // 剪枝:当前选中的数大于m, 当前选中的数+剩余未选中的数小于m
    if(path.size() > m || path.size() + (n - u + 1) < m) return; 
    if(u == n + 1){
        int sum = 0;
        // 对组合出来的下标带入原数组求值
        for(int i = 0; i < path.size(); i ++){
            int idx = path[i];
            // cout << idx << endl;
            sum += a[idx];
        }
        // cout << "sum" << sum << endl;
        if(isprim(sum)) cnt ++;
        return; 
    }

    // 不选择数u分支
    dfs(u + 1);

    // 选择数u分支
    path.push_back(u);
    dfs(u + 1);

    // 恢复现场
    path.pop_back();
}
void solve(){
    cin >> n >> m;
    for(int i = 1; i <= n; i ++) cin >> a[i];
    dfs(1);
    cout << cnt;
}
int main(){
    io_speed
    solve();
    return 0;
}