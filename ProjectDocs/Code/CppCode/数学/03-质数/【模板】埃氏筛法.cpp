// https://www.luogu.com.cn/problem/P3383
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
const int N = 1e8 + 10;
int vis[N];  // 真为合数
int prim[N]; // 记录质数
int cnt;     // 质数个数
void Erato(int n){ // 埃氏筛
    for(LL i = 2; i <= n; i ++){
        if(!vis[i]){ // 当前数没有被划掉，一定是质数
            prim[++ cnt] = i;
            // 划掉改该质数的倍数,一定是合数
            // 从 i * 2 - i * （i - 1) 在这之前就被划掉了,注意LL +=i
            for(LL j = i * i; j <= n; j += i) vis[j] = 1;
        }
    }
}
void solve(){
    int n, q; cin >> n >> q;
    Erato(n);
    while(q --){
        int k; cin >> k;
        cout << prim[k] << endl;
    }

}
int main(){
    io_speed
    solve();
    return 0;
}