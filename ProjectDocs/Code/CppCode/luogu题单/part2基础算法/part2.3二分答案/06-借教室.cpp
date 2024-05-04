// https://www.luogu.com.cn/problem/P1083
// 对于m份订单，假设第i份不能满足，那么后面的都不能满足，具有二段性，可以用二分来做
// 对于check函数, 采用差分和前缀和判断
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
int n, m;
int r[N], d[N], s[N], t[N], backup[N], b[N];
// check函数如果是暴力做的话，也会超时，因为是在区间上进行加减操作，不妨试一下差分
// bool check(int x){  // 检验当订单数为1~x时是否可以满足要求
//     memcpy(backup, r, sizeof r);
//     for(int i = 1; i <= x; i ++){
//         int st = s[i], ed = t[i], dd = d[i];
//         for(int j = st; j <= ed; j ++){
//             backup[j] -= dd;
//             if(backup[j] < 0) return false;
//         }
//     }
//     return true;
// }


bool check(int x){
    memset(b, 0, sizeof b);
    // 求数组r[i]的差分序列
    for(int i = 1; i <= n; i ++) b[i] = r[i] - r[i - 1];

    // 对原数组区间加减操作->对差分数组的两点操作
    for(int i = 1; i <= x; i ++){
        int st = s[i], ed = t[i], dd = d[i];
        b[st] += -dd;
        b[ed + 1] -= -dd;
    }
    // 前缀和
    for(int i = 1; i <= n; i ++){
        b[i] += b[i - 1];
        if(b[i] < 0) return false;
    }
    return true;

}
// 二分最大的满足要求的m, 超过m的订单就不满足了
int find(int l, int r){
    while(l + 1 < r){
        int mid = l + r >> 1;
        if(check(mid)) l = mid;
        else r = mid;
    }
    return l;
}
void solve(){
    cin >> n >> m;
    for(int i = 1; i <= n; i ++) cin >> r[i];
    for(int i = 1; i <= m; i ++) cin >> d[i] >> s[i] >> t[i];


    // 先判断前m个订单是否都满足
    if(check(m)) cout << 0;
    else{
        int l = 1 - 1, r = m + 1;
        int ans = find(l, r);
        cout << -1 << endl;
        cout << ans + 1; // ans是满足的订单编号 ans + 1是不满足的
    }
    
}
int main(){
    io_speed
    solve();
    return 0;
}
