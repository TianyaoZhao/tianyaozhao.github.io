// https://www.acwing.com/problem/content/3712/
// 给定单链表,交换相邻两个节点,如果是奇数,不用交换
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
int a[N];
int n;
void solve(){
    cin >> n;
    for(int i = 1; i <= n; i ++) cin >> a[i];
    int cnt = 0;
    if(n % 2 == 0) cnt = n;
    else cnt = n - 1;
    for(int i = 1; i <= cnt; i += 2){
        swap(a[i], a[i + 1]);
    }
    for(int i = 1; i <= n; i ++) cout << a[i] << " ";
}
int main(){
    io_speed
    solve();
    return 0;
}