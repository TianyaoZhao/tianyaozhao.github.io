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
void dfs(int x){
    if(x == 0) cout << "2(0)";
    else if(x == 1) cout << "2";
    else if(x == 2) cout << "2(2)";
    else{
        int t = x;
        int cnt = - 1;
        while(t){
            cnt ++;
            if(t & 1){
                dfs(cnt);
        }
    }
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