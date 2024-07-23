// https://www.acwing.com/problem/content/3396/
// 求连续子序列的最大和
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
const int N = 1e6 + 10;
LL f[N];   // f[i]表示以到ai结尾的最大和
LL a[N], n;
void solve(){
    cin >> n;
    for(int i = 1; i <= n; i ++){
        cin >> a[i];
        f[i] = a[i]; //初始化
    }
    LL ans = -1e9;
    for(int i = 1; i <= n; i ++){
        f[i] = max(f[i], f[i - 1] + a[i]);
        ans = max(ans, f[i]);
    }
    cout << ans;
}
int main(){
    io_speed
    solve();
    return 0;
}