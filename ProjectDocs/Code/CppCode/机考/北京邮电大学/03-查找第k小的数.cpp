// https://www.acwing.com/problem/content/3536/
// 查找一个长度为 n 的数组中第 k 小的数。实际上就是快速排序，相同大小算一样大，去重
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
int n, k;
vector <int> a;
void solve(){
    cin >> n;
    for(int i = 0; i < n; i ++){
        int x; cin >> x;
        a.push_back(x);
    }
    cin >> k;
    sort(a.begin(), a.end());
    // 去重
    unique(a.begin(), a.end());
    cout << a[k - 1];
}
int main(){
    io_speed
    solve();
    return 0;
}