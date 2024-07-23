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
int f[N][N];     // 前i件物品,背包容量为j时的最大价值
void solve(){
    cin >> n >> m; 
    for(int i = 1; i <= n; i ++) cin >> w[i] >> v[i];

    for(int i = 1; i <= n; i ++){
        for(int j = 1; j <= m; j ++){
            if(j < w[i]) f[i][j] = f[i - 1][j];
            else f[i][j] = max(f[i - 1][j], f[i][j - w[i]] + v[i]);  // 第i件物品不放入背包和放入背包
        }
    }
    cout << f[n][m];
}
int main(){
    io_speed
    solve();
    return 0;
}