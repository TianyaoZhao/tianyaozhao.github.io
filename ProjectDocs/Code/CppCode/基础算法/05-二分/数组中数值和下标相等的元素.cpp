// https://www.acwing.com/problem/content/65/
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
const int N = 100 + 10;
int n, a[N];

void solve(){
    cin >> n;
    for(int i = 0; i < n; i ++) cin >> a[i];
    
    // 找到第一个下标等于数值的数
    int l = -1, r = n;
    while(l + 1 < r){
        int mid = l + r >> 1;
        if(a[mid] >= mid) r = mid;
        else l = mid;
    }
    int st = r;
    l = 0, r = n + 1;
    while(l + 1 < r){
        int mid = l + r >> 1;
        if(a[mid] <= mid) l = mid;
        else r = mid;
    }
    int ed = l;
    if(a[st] != st && a[ed] != ed){
        cout << -1;
    }
    else{
        for(int i = st; i <= ed; i ++){
            cout << a[i] << " ";
        }
    }
}
int main(){
    io_speed
    solve();
    return 0;
}