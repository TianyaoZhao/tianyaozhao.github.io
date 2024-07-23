// http://ybt.ssoier.cn:8088/problem_show.php?pid=1221
// 给定n个正整数，将他们分组，使得每组中任意两个数互为质数，至少分成几个组
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
const int N = 10 + 10;
int n, a[N];
int ans = N, cnt;
vector <int> g[N];
int gcd(int a, int b){
    return b ? gcd(b, a % b) : a;
}
bool check(int x, int i){
    for(int j = 0; j < g[i].size(); j ++){
        // x和当前组内的数有公约数，就不互质
        if(gcd(x, g[i][j]) > 1) return false; 
    }
    return true;
}
void dfs(int u){
    if(cnt >= ans) return;          // 当前分的组数大于已知的答案，剪枝
    if(u == n){                     // 找到一组方案
        ans = min(ans, cnt);
        return; 
    }
    // 枚举现有的组
    for(int i = 0; i < cnt; i ++){
        // 如果和当前组互质
        if(check(a[u], i)){
            // 互质就放入当前组
            g[i].push_back(a[u]);
            dfs(u + 1);
            // 恢复现场
            g[i].pop_back();
        }
    }
    // 已有的组不满足互质条件, 新开一组
    g[cnt ++].push_back(a[u]); 
    dfs(u + 1);
    // 恢复现场
    g[-- cnt].pop_back();
     
}
void solve(){
    cin >> n;
    for(int i = 0; i < n; i ++) cin >> a[i];
    dfs(0);
    cout << ans;
}   
int main(){
    io_speed
    solve();
    return 0;
}