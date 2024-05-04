// https://www.luogu.com.cn/problem/P1226
// 快速幂采用二进制拆分和倍增思想
// 3^{13} = 3^{(1101)_2} = 3^8·3^4·3^1
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
// const int N = 
LL a, b, p;
LL quickpow(LL a, LL b, LL p){
    // 如果b是偶数, 先计算 a^(b / 2)次方，直到计算到 a^1
}
void solve(){
    cin >> a >> b >> p;
    cout << a << "^" << b << " mod " << p << "=" << quickpow(a, b, p);
}
int main(){
    io_speed
    solve();
    return 0;
}