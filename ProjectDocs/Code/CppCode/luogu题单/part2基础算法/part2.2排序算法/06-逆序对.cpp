// https://www.luogu.com.cn/problem/P1908
# include <iostream>
# include <cstring>
# include <algorithm>
# include <string>
# include <cmath>
# include <stack>
# include <queue>
# include <vector>
# define io_speed ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
# define endl '\n'
using namespace std;
typedef pair <int, int> PII;
typedef long long LL;
const int N = 5e5 + 10;
int a[N], b[N], n;
LL ans;
void msort(int l, int r){
    if(l == r) return;
    int mid = l + r >> 1;
    msort(l, mid);
    msort(mid + 1, r);

    int i = l , j = mid + 1, k = l;
    while(i <= mid && j <= r){
        if(a[i] <= a[j]) b[k ++] = a[i ++];
        else b[k ++] = a[j ++], ans += mid - i + 1; 
    }
    while(i <= mid) b[k ++] = a[i ++];
    while(j <= r) b[k ++] = a[j ++];
    for(int i = l; i <= r; i ++) a[i] = b[i];
}
void solve(){
    cin >> n;
    for(int i = 0; i < n; i ++) cin >> a[i];
    msort(0, n - 1);
    cout << ans;
}
int main(){
    io_speed
    solve();
    return 0;
}