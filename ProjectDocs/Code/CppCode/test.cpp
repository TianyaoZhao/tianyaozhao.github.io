# include <iostream>
# include <cstring>
# include <algorithm>
# include <string>
# include <cmath>
# include <stack>
# include <queue>
# include <vector>
# define io_speed ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
# define endl '\n'
using namespace std;
typedef pair <int, int> PII;
typedef long long LL;
const int N = 20 + 10;
int n, m;
vector <int> path;
void dfs(int u){
    //  剪枝:当前选的数大于m 或 当前选的数+剩下没选的数小于m
    if(path.size() > m || path.size() + (n - u + 1) < m) return;
    if(u == n + 1){ // 问题的边界
        for(int i = 0; i < path.size();  i ++){
            cout << path[i];
        }
        cout << endl;
        return;
    }
    // 不选数u分支
    dfs(u + 1);

    // 选数u分支
    path.push_back(u);
    dfs(u + 1);

    // 恢复现场
    path.pop_back();
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
