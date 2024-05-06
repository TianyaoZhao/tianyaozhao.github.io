// https://www.luogu.com.cn/problem/P1902
// 看到最小字眼，就是二分了
// 二分加搜索（check），二分最大伤害，搜索时只有p[i][j]小于等于最大伤害的可以走；
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
int p[N][N];
int dx[5] = {0, 1, -1, 0}, dy[5] = {1, 0, 0, -1};
int n, m; 
bool flag, vis[N][N];
void dfs(int x, int y, int mid){
    // 只保留了没有超出边界且伤害小于等于mid，且没有走过的节点
    if(x < 1 || x > n || y < 1 || y > m || p[x][y] > mid || vis[x][y] == true) return;

    vis[x][y] = true;
    // 已经有答案了不用继续搜了
    if(flag) return;
    if(x == n){
        // 找到这样的路了
        flag = true;
        return;
    }
    
    // 枚举四个方向
    for(int i = 0; i < 4; i ++){
        int nx = x + dx[i];
        int ny = y + dy[i];
        dfs(nx, ny, mid);
    }
}
bool check(int mid){ // 表示能在小于等于mid伤害下找到一条路
    memset(vis, 0, sizeof vis);
    flag = false;
    dfs(1, 1, mid);  // 从(1, 1)开始搜
    return flag;
}
int find(int l, int r){
    while(l + 1 < r){
        int mid = l + r >> 1;
        if(check(mid)) r = mid; // 说明最小伤害还要小， mid大了
        else l = mid;           // 说明最小伤害还要大， mid小了
    }
    return l + 1;
}
void solve(){
    cin >> n >> m;
    int maxn = 0, minn = 10000;
    for(int i = 1; i <= n; i ++){
        for(int j = 1; j <= m; j ++){
            cin >> p[i][j];
            maxn = max(maxn, p[i][j]);
            if(p[i][j] != 0) minn = min(minn, p[i][j]);
        }
    }

    int l = minn - 1, r = maxn + 1;
    // 二分最大伤害代价
    int ans = find(l, r);
    cout << ans;
}
int main(){
    io_speed
    solve();
    return 0;
}