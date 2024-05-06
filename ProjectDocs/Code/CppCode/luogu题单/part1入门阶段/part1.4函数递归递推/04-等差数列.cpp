// https://www.luogu.com.cn/problem/P5534
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
// const int N = 
LL a1, a2, n, d, sum;
void solve(){
    cin >> a1 >> a2 >> n;
    d = a2 - a1;
    for(int i = 1; i <= n; i ++){
        LL a = a1 + (i - 1) * d;
        sum += a;
    }
    cout << sum;
}
int main(){
    io_speed
    solve();
    return 0;
}