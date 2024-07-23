// https://www.nowcoder.com/practice/a5a0b05f0505406ca837a3a76a5419b3?tpId=61&tqId=29535&tPage=2&ru=/kaoyan/retest/1002&qru=/ta/pku-kaoyan/question-ranking
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
int n, a[N][N], s[N][N]; // 前缀和预处理
int f[N][N]; // f[i][j] 表示以a[i][j]结尾的最大子矩阵和
int query(int x1, int y1, int x2, int y2){
    return s[x2][y2] - s[x2][y1 - 1] - s[x1- 1][y2] + s[x1 -1][y1 - 1]; 
}
void solve(){
    cin >> n;
    for(int i =1; i <= n; i ++){
        for(int j = 1; j <= n; j ++){
            cin >> a[i][j];
            f[i][j] = a[i][j];
            s[i][j] = s[i][j - 1] + s[i - 1][j] - s[i - 1][j - 1] + a[i][j];
        }
    }
    int ans = -1e6;
    for(int i = 1; i <= n; i ++){
        for(int j = 1; j <= n; j ++){
            f[i][j] = max(f[i][j], f[i - 1][j] + f[i][j - 1] - f[i - 1][j - 1] + a[i][j]);
            ans = max(ans, f[i][j]);
        }
    }
    cout << ans;
}
int main(){
    io_speed
    solve();
    return 0;
}