// 输入123，输出321
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
// const int N = 
int n;
void solve(){
    cin >> n;
    int ans = 0; // 答案
    while(n > 0){
        ans = ans * 10;
        ans += n % 10;
        n = n / 10;
    }
    cout << ans;
}
int main(){
    io_speed
    solve();
    return 0;
}