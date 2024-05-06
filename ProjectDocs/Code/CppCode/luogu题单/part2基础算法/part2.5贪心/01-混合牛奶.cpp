// https://www.luogu.com.cn/problem/P1208
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
const int N = 5000 + 10;
int n, m;
struct sell{
    int val;
    int num;
}a[N];
bool cmp(sell aa, sell bb){
    return aa.val < bb.val;
}
void solve(){
    cin >> n >> m;
    for(int i = 0; i < m; i ++) cin >> a[i].val >> a[i].num;
    sort(a, a + m, cmp);
    int tol = 0, pay = 0;
    for(int i = 0; i < m; i ++){
        if(tol < n){
            // 加上当前的数量不超
            if(tol + a[i].num <= n){
                tol += a[i].num;
                pay += a[i].val * a[i].num;
            }
            // 加上当前的数量超了
            else{
                pay += a[i].val * (n - tol);
                break;
            }
        }
        if(tol == n) break;
    }
    cout << pay;
}
int main(){
    io_speed
    solve();
    return 0;
}