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
const int N = 300 + 10;
int n, a[N], s[N], f[N][N];
void solve(){
    cin >> n;
    for(int i = 1; i <= n; i ++){
        cin >> a[i];
        s[i] = s[i - 1] + a[i];
    }
    
    // 枚举区间长度
    for(int len = 2; len <= n; len ++){
        // 枚举起点
        for(int i = 1; i + len - 1 <= n; i ++){
            int l = i, r = i + len - 1;
            // 枚举分界
            f[l][r] = 1e8;
            for(int k = l; k <= r - 1; k ++){
                f[l][r] = min(f[l][r], f[l][k] + f[k + 1][r] + s[r] - s[l - 1]);
            }
        }
    }
    cout << f[1][n];
}
int main(){
    io_speed
    solve();
    return 0;
}