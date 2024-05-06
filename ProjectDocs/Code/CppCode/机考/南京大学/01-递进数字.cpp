// acwing.com/problem/content/3713/
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
const int N = 20;
int T, a[N], cnt;
void check(int x){

    int k = 0;
    memset(a, 0, sizeof a);
    while(x){
        int t = x % 10;
        x = x / 10;
        a[k ++] = t;
    }

    bool flag = false;
    for(int i = 1; i < k; i ++){
        if(abs(a[i] - a[i - 1]) != 1){
            flag = true;
            break;
        }
    }
    if(!flag) cnt ++;
}
void solve(){
    cin >> T;
    while(T --){
        int l, r;
        cin >> l >> r;
        cnt = 0;
        for(int i = l; i <= r; i ++){
            if(i < 10) continue;
            check(i);
        }
        cout << cnt << endl;
    }
}
int main(){
    io_speed
    solve();
    return 0;
}