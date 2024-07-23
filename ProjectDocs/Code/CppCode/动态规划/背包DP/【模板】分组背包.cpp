// https://www.acwing.com/problem/content/9/
// 每组物品有若干类，同一组内的物品最多只能选一类
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
const int N = 100 + 10;
int w[N][N], v[N][N]; // 第i组第j个物品的重量和价值
int s[N];             // 每组的物品种类数
int n, m;             // 物品组数背包容量
int f[N][N];          // 前i组物品背包容量为j时的最大价值
void solve(){
    cin >> n >> m;
    for(int i = 1; i <= n; i ++){
        cin >> s[i];
        for(int j = 1; j <= s[i]; j ++){
            cin >> w[i][j] >> v[i][j];
        }
    }

    for(int i = 1; i <= n; i ++){ // 枚举组数
        for(int j = 1; j <= m; j ++){ // 枚举容量
            // 不选第i组
            f[i][j] = f[i - 1][j];
            // 选第i组
            for(int k = 1; k <= s[i]; k ++){ // 枚举所有选择
                if(w[i][k] <= j){
                    f[i][j] = max(f[i][j], f[i - 1][j - w[i][k]] + v[i][k]);
                }
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