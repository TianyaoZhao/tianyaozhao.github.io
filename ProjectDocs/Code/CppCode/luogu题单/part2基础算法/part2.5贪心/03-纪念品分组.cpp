// https://www.luogu.com.cn/problem/P1094
// 贪心策略：每次先取最贵的，然后看是否再放一件最便宜的
// 不能放，就单独打包，能放就合并打包
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
const int N = 3e4 + 10;
int w, n, p[N];
void solve(){
    cin >> w >> n;
    for(int i = 0; i < n; i ++) cin >> p[i];
    sort(p, p + n);
    int maxn = n - 1, minn = 0;
    int cnt = 0;
    // 循环结束存在两种可能
    // 1.maxn = minn 表示最后剩下一种物品单独打包
    // 2.maxn < minn 表示最后没有物品可以打包
    while(maxn > minn){
        // 能放进去最便宜的
        if(p[maxn] + p[minn] <= w){
            cnt ++;
            maxn --;
            minn ++;
        }
        // 不能放进去最便宜的
        else{
            cnt ++;
            maxn --;
        }
    }
    if(maxn == minn) cnt ++;
    cout << cnt ;
}
int main(){
    io_speed
    solve();
    return 0;
}