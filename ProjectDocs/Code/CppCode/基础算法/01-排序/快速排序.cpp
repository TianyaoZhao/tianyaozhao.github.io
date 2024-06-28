// https://www.acwing.com/file_system/file/content/whole/index/content/4313/
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
const int N = 1e5 + 10;
int n, a[N];
void qsort(int l, int r){
    if(l == r) return;
    int i = l - 1, j = r + 1, x = a[l + r >> 1];
    while(i < j){
        do i ++; while(a[i] < x);    // 向右找>=x的数
        do j --; while(a[j] > x);    // 向左找<=x的数
        if(i < j) swap(a[i], a[j]);  // i在左j在右
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