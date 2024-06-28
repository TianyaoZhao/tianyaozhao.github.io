// https://www.acwing.com/problem/content/3704/
// 求ab之间合数的个数
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
const int N = 1e7 + 10;
bool vis[N];      // true为合数
int prim[N], cnt; // 记录质数
int s[N];         // 前缀和
int n;
void etos(){
    for(LL i = 2; i < N; i ++){
        if(!vis[i]){
            prim[++ cnt] = i; //记录质数
            for(LL j = i * i; j < N; j += i){ // 划掉合数
                vis[j] = true;
            }
        }
    }
}
void solve(){
    etos(); // 处理质数
    // 前缀和处理
    for(int i = 1; i < N; i ++){
        if(vis[i]) s[i] = s[i - 1] + 1;
        else s[i] =  s[i - 1];
    }
    int a, b;
    while(cin >> a >> b){
        cout << s[b] - s[a - 1] << endl;
    }
}
int main(){
    io_speed
    solve();
    return 0;
}