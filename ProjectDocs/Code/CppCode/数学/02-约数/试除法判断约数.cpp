// https://www.acwing.com/activity/content/problem/content/938/
// a是n的约数，那么n/a也是约数，最多只有一个大于sqrt(n)的约数
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
const int N = 1e6 + 10;
int a[N]; //记录约数
int cnt;  //约数个数
void get_disvisor(int x){
    for(int i = 1; i <= x / i; i ++){
        if(x % i == 0){
            // 两个约数
            a[cnt ++] = i;
            if(i != x / i) a[cnt ++] = x / i; 
        } 
    }
}
void solve(){
    int n; cin >> n;
    while(n --){
        int x; cin >> x;
        get_disvisor(x);
        sort(a, a + cnt);
        for(int i = 0; i < cnt; i ++) cout << a[i] << " ";
        cout << endl;
        memset(a, 0, sizeof a);
        cnt = 0;
    }
}
int main(){
    io_speed
    solve();
    return 0;
}