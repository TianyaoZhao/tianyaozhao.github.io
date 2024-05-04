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
void dfs(int n){
    int t = n;
    int cnt  = 0;
    while(t){
        if(t & 1) cnt ++;
    }
}
void solve(){
    cin >> n;
    dfs(n);
}
int main(){
    io_speed
    solve();
    return 0;
}