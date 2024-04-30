// https://www.luogu.com.cn/problem/P1177
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
const int N = 1e5 + 10;
int a[N], n;
void qsort(int l, int r){
    if(l == r) return;
    int i = l - 1, j = r + 1, x = a[l + r >> 1];
    while(i < j){
        do i ++; while(a[i] < x);
        do j --; while(a[j] > x);
        if(i < j) swap(a[i], a[j]);
    }
    qsort(l, j);
    qsort(j + 1, r);
}
void solve(){
    cin >> n;
    for(int i = 0; i < n; i ++) cin >> a[i];
    qsort(0, n - 1);
    for(int i = 0; i < n; i ++) cout << a[i] << " "; 
}
int main(){
    io_speed
    solve();
    return 0;
}