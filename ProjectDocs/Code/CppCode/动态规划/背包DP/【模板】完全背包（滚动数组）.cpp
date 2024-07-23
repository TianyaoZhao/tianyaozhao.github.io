// https://www.acwing.com/problem/content/3/
// 每种物品有无数多个，可以一直往背包装
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
const int N = 1000 + 10;
int n, m;        // 数量和容量
int v[N], w[N];  // 价值和体积
int f[N];        // 前i件物品,背包容量为j时的最大价值
void solve(){
    cin >> n >> m; 
    for(int i = 1; i <= n; i ++) cin >> w[i] >> v[i];

    for(int i = 1; i <= n; i ++){
        for(int j = w[i]; j <= m; j ++){     // 顺序循环,防止越界，数组下标j从w[i]开始,小于w[i]的都是上层的值
            f[j] = max(f[j], f[j - w[i]] + v[i]);
        }
    }
    cout << f[m];
}
int main(){
    io_speed
    solve();
    return 0;
}