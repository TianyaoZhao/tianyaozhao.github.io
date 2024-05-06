// https://www.luogu.com.cn/problem/P1199
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
const int N = 500 + 10; 
int n;
int v[N][N], vis[N];
void solve(){
    cin >> n;
    for(int i = 1; i <= n; i ++){
        for(int j = 1; j <= n; j ++){
            if(i >= j) continue;
            cin >> v[i][j];
            v[j][i] = v[i][j];
        }
    }

    // for(int i = 1; i <= n; i ++){
    //     for(int j = 1; j <= n; j ++){
    //         cout << v[i][j] << ' ';
    //     }
    //     cout << endl;
    // }
    
    // 起手选武将有i种选择

    for(int i = 1; i <= n; i ++){
        memset(vis, 0, sizeof vis);
        
    }
}
int main(){
    io_speed
    solve();
    return 0;
}