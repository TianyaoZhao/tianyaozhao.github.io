// https://www.luogu.com.cn/problem/P4995
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
const int N = 300 + 10;
LL n, h[N];
void solve(){
    cin >> n;
    for(int i = 1; i <= n; i ++) cin >> h[i];
    sort(h, h + n + 1);
    int i = 0, j = n;
    LL tot = 0, step = i;
    while(i != j){
        if(step < j){
            tot += (h[j] - h[step]) * (h[j] - h[step]);
            step = j;
            i ++;
        } 
        if(step > i){
            tot += (h[i] - h[step]) * (h[i] - h[step]);
            step = i;
            j --;
        }
    }
    cout << tot;
}
int main(){
    io_speed
    solve();
    return 0;
}
