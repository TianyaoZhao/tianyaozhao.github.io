// http://noi-test.zzstep.com/contest/0x20%E3%80%8C%E6%90%9C%E7%B4%A2%E3%80%8D%E4%BE%8B%E9%A2%98/2201%20%E5%B0%8F%E7%8C%AB%E7%88%AC%E5%B1%B1
// n只小猫，重量为c[N]，每个缆车不能超重w，问最少需要几个缆车
# include <iostream>
# include <cstring>
# include <algorithm>
# include <string>
# include <cmath>
# include <stack>
# include <queue>
# include <vector>
# include <iomanip>
# define io_speed ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
# define endl '\n'
using namespace std;
typedef pair <int, int> PII;
typedef long long LL;
const int N = 18 + 10;
int c[N], n, w;
int sum[N], cnt; // 当前的缆车数量以及第i个缆车上小猫的总重量
int ans = N;     // 最优解
bool check(int u, int i){
    if(sum[i] + c[u] <= w) return true;
    else return false;
}
void dfs(int u){
    if(cnt >= ans) return; // 最优性剪枝
    if(u == n) {ans = min(ans, cnt); return;} // 找到一个方案

    // 遍历每个车
    for(int i = 0; i < cnt; i ++){
        if(check(u, i)){ // 当前车符合要求
            sum[i] += c[u];
            dfs(u + 1);
            // 恢复现场
            sum[i] -= c[u];
        }
    }

    // 如果不满足要求就重开一辆车
    sum[cnt ++] += c[u];
    dfs(u + 1);
    // 恢复现场
    sum[-- cnt] -= c[u]; 

}
void solve(){
    cin >> n >> w;
    for(int i = 0; i < n; i ++) cin >> c[i];
    // 剪枝优化搜索顺序，大的先搜索
    sort(c, c + n);
    reverse(c, c + n);
    dfs(0);
    cout << ans;
}
int main(){
    io_speed
    solve();
    return 0;
}