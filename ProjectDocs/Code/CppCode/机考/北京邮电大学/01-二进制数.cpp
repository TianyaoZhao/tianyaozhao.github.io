// https://www.acwing.com/problem/content/3533/
// 输入unsigned int 输出去除前导0的二进制串
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
unsigned int n;
char s[N];
int k;
void ten2x(unsigned int n, int x){
    if(n == 0) cout << 0;
    memset(s, 0, sizeof s);
    k = 0;
    while(n > 0){
        int w = n % x;
        if(w < 10) s[k ++] = w + '0';
        else s[k ++] = w - 10 + 'A';
        n = n / x;
    }
    for(int i = k - 1; i >= 0; i --){
        cout << s[i];
    }
    cout << endl;
}
void solve(){
    while(cin >> n){
        ten2x(n, 2);
    }
}
int main(){
    io_speed
    solve();
    return 0;
}