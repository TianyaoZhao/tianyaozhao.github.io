// https://www.acwing.com/problem/content/3725/
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
const int N = 1000 + 10;
int a[N], n, maxn; 
void solve(){
    while(cin >> n){
        for(int i = 1; i <= n; i ++) cin >> a[i];
        
        int cnt= 0;
        for(int i = 2; i <= n; i ++){
            if(a[i] >= a[i - 1]){
                cnt ++;
            }
            else{ // 出现小于的
                if(cnt >= 1){
                    maxn = max(maxn,  a[i - 1] - a[i - 1 - cnt]);
                }
                cnt = 0; 
            }
        }
        // 单独处理一下最后的合法数据
        maxn = max(maxn, a[n] - a[n - cnt]);
        cout << maxn << endl;
        maxn = 0;
    }
}
int main(){
    io_speed
    solve();
    return 0;
}