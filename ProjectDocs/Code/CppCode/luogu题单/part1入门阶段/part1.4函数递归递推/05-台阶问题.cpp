// https://www.luogu.com.cn/problem/P1192
# include <iostream>
# include <cstring>
# include <algorithm>
# include <string>
# include <cmath>
# include <stack>
# include <queue>
# include <vector>
# define io_speed ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
# define endl '\n'
using namespace std;
typedef pair <int, int> PII;
typedef long long LL;
const int MOD = 1e5 + 3;
<<<<<<< HEAD
const int N = 1e5 + 10
int n, k;
LL f[N]; // f[i]表示到第i个台阶可能的情况
void solve(){
    cin >> n >> k;
    f[1] = 1;
    for(int i = 2; i <= n; i ++){
        for(int j = 1; j <= min(i, k); j ++){
            f[i] += f[i - j];
        }
    }
    cout << f[n] % MOD;
=======
const int N = 1e5 + 10;
int n, k;
int f[N]; // f[i]表示到第i个台阶可能的情况
void solve(){
    cin >> n >> k;
    f[0] = 1;
    for(int i = 1; i <= n; i ++){
        for(int j = 1; j <= min(i, k); j ++){
            f[i] = (f[i] + f[i - j]) % MOD;
        }
    }
    cout << f[n];
>>>>>>> e221473eb78db730f77ae49922bd65c0b168e7f7
}
int main(){
    io_speed
    solve();
    return 0;
}