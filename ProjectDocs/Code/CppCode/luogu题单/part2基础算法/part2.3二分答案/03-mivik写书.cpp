// https://www.luogu.com.cn/problem/P1316
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
vector <int> path;
int n, m;
int calc(vector <int> path){

}
void dfs(int u){
    if(u == m + 1){
        for(int i = 0; i < path.size(); i ++){
            cout << path[i] << " ";
        }
        cout << endl;
        // calc(path);
        return;
    }
    for(int i = 1; i <= n; i ++){
        path.push_back(i);
        dfs(u + 1);
        path.pop_back();
    }
}
void solve(){
    cin >> n >> m;
    dfs(1);
}
int main(){
    io_speed
    solve();
    return 0;
}