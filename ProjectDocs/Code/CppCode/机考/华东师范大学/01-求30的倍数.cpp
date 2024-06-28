// https://www.acwing.com/problem/content/3724/
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
string s;
int cnt[N];
int l, maxn;
void dfs(int u, int s){
    if(u == l + 1 && s % 30 == 0){
        maxn = max(maxn, s);
        return;
    }

    // 遍历每个数
    for(int i = 0; i < 10; i ++){
        if(cnt[i]){ // 当前数还可用
            cnt[i] --;
            // 递归找下一个数
            dfs(u + 1, s * 10 + i);
            // 恢复现场
            cnt[i] ++;
        }
    }
    

}
void solve(){
    cin >> s;
    // 统计每个数出现的次数
    l = s.length();
    for(int i = 0; i < l; i ++) cnt[s[i] - '0'] ++;
    // 枚举第i个位置
    dfs(1, 0);
    if(maxn == 0) cout << -1;
    else cout << maxn;



}
int main(){
    io_speed
    solve();
    return 0;
}