// https://leetcode.cn/problems/reverse-linked-list/description/
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
const int N = 5000 + 10;
int a[N], n;
void dfs(int u){
    if(u == n){
        return;
    }
    dfs(u + 1);
    cout << a[u] << endl;
}
void solve(){
    cin >> n;
    for(int i = 0; i < n; i ++){
        cin >> a[i];
    }
    dfs(0);
}
int main(){
    io_speed
    solve();
    return 0;
}