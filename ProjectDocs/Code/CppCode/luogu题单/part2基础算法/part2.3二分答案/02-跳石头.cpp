// https://www.luogu.com.cn/problem/P2678
// N块石头，搬走M块，求两个石头之间最短距离的最大值
// 思路：
// 1.拿走的石头越多，最短的跳跃距离越大，符合单调性
// 2.从a[1]~l二分最短跳跃距离，这个距离需要满足：
// 3.在当前最短跳跃距离x下，最多有m块石头的相邻距离要小于等于x
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
const int N = 5e4 + 10;
int s, n, m;
int a[N];
bool check(int mid){
    // last上一块石头的位置
    int last = 0, cnt = 0; 
    for(int i = 1; i <= n + 1; i ++){ // 枚举每一块石头
        if(a[i] - last < mid) cnt++;  // 表示当前距离小于最短距离，当前石头需要移走
        else last = a[i];             // 表示当前距离大于等于最短距离，当前石头无需移走
    }
    if(cnt > m) return false;         // 移走的石头数目超过m
    else return true;
}
int find(int l, int r){
    // 当前的最短相邻距离是合法的
    while(l + 1 < r){
        int mid = l + r >> 1;
        if(check(mid)) l = mid;
        else r = mid;
    }
    return l;
}
void solve(){
    cin >> s >> n >> m;
    for(int i = 1; i <= n; i ++) cin >> a[i];
    a[n + 1] = s;
    // 指向二分的答案区间的外侧
    int l = 1 - 1, r = s + 1;
    int ans = find(l, r);
    cout << ans;

}
int main(){
    io_speed
    solve();
    return 0;
}