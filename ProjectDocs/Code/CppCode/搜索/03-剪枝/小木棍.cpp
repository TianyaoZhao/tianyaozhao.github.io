// 等长小木棍，任意砍几段，直到每段不超过50。现在想恢复原样，但是不知道有多少根
// 也不知道原先多长，给出每段小木棍的长度，找出最小可能长度
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
const int N = 50 + 10;
int n, a[N];
void dfs(int u){
    
} 
void solve(){
    cin >> n;
    for(int i = 0; i < n; i ++) cin >> a[i];
    // 从小到大枚举长度len，总长sum必须是len的整数倍(cnt)
    // 搜索所有小木棍，看看能不能拼凑成cnt个长度为len的木棍
    dfs(0);
}
int main(){
    io_speed
    solve();
    return 0;
}