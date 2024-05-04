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
    LL res = 1;
    while(b){
        if(b & 1) res = res * a % p;  // 二进制1对应位相乘
        a = a * a % p;                // 倍增
        b >>= 1;
    }
    return res;
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