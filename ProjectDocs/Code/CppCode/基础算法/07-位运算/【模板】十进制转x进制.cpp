// 将10进制转为x进制，通用版
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
const int N = 100;
int n, x;  // 数，进制
int k;     // 数组下标
char s[N]; // 字符串存储最终结果
void ten2x(int n, int x){
    while(n > 0){ 
        int w = n % x;                 // 按位分解
        if(w < 10) s[k ++] = w + '0';  // 变成字符需要+'0'
        else s[k ++] = (w - 10) + 'A'; // 大于9要变字母
        n = n / x;
    }
    // 反序输出
    for(int i = k - 1; i >= 0; i --) cout << s[i];
}
void solve(){
    cin >> n >> x;
    ten2x(n, x);
}
int main(){
    io_speed
    solve();
    return 0;
}