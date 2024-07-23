// https://www.acwing.com/problem/content/3/
// 每种物品有有限多个，可以一直往背包装
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
int n, m;              // 数量和容量
int v[N], w[N], s[N];  // 价值和体积,每种物品的数量
int f[N][N];           // 前i件物品,背包容量为j时的最大价值
void solve(){
    cin >> n >> m; 
    for(int i = 1; i <= n; i ++) cin >> w[i] >> v[i] >> s[i];

    for(int i = 1; i <= n; i ++){
        for(int j = 1; j <= m; j ++){
           for(int k = 0; k <= s[i]; k ++){
                if(k * w[i] > j) f[i][j] = f[i][j];                              //当前装不下k件物品
                else f[i][j] = max(f[i][j], f[i - 1][j - k * w[i]] + k * v[i]);  //当前可装下k件物品
           }
        }
    }
    cout << f[n][m];
}
int main(){
    io_speed
    solve();
    return 0;
}