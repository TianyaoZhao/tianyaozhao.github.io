// https://www.luogu.com.cn/problem/P5736
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
const int N = 100 + 5;
int n;
bool isprime(int x){
    if(x == 1) return false; // 注意1既不是质数也不是合数
    for(int i = 2; i <= x / i; i ++){
        if(x % i == 0) return false;
    }
    return true;
}
void solve(){
    cin >> n;
    int x;
    for(int i = 0; i < n; i ++){
        cin >> x;
        if(isprime(x)) cout << x << " ";
    }
}
int main(){
    io_speed
    solve();
    return 0;
}