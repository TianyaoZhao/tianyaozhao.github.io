// https://www.acwing.com/activity/content/problem/content/923/
/* 给定一个 n个点m条边的有向图，图中可能存在重边和自环，边权可能为负数 */
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
const int N = 200 + 10;
int d[N][N];
int n, m, k;
void floyd(){
    // k放在最外层,因为下一层的状态需要等上一层完全计算出来
    for(int k = 1; k <= n; k ++){
        for(int i = 1; i <= n; i ++){
            for(int j = 1; j <= n; j ++){
                d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
            }
        }
    }
}
void solve(){
    cin >> n >> m >> k;
    memset(d, 0x3f, sizeof d);
    while(m --){
        int u, v, w;
        cin >> u >> v >> w;
        d[u][v] = min(d[u][v], w);
    }
    // d[i][i] = 0
    for(int i = 1; i <= n; i ++) d[i][i] = 0;
    floyd();
    while(k --){
        int x, y;
        cin >> x >> y;
        if(d[x][y] < 0x3f3f3f3f / 2) cout << d[x][y] << endl; 
        else cout << "impossible" << endl;
    }
}
int main(){
    io_speed
    solve();
    return 0;
}