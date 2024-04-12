// dfs(u) + 判重数组vis
# include <iostream>
# include <cstring>
# include <algorithm>
using namespace std;
const int N = 300 + 10;
int n, m;      // n个点m条边
vector <int> e[N];   // 链式邻接表,e[u]存放与u点邻接的点
bool vis[N];

void dfs(int u){
    // 入:进入父节点
    // 遍历u的每一条出边
    vis[u] = true;
    for(auto v : e[u]){
        if(vis[v]) continue; // 防止回到已经搜过的节点, 重复遍历
        // 下：从父节点进入子节点
        cout << u << "->" << v << endl;
        dfs(v);
        // 回：从一个子节点回到父节点
    }
    // 离：从所有子节点离开到父节点
}
int main(){
    cin >> n >> m;
    for(int i = 0; i < m; i ++){
        int a, b;
        cin >> a >> b;
        e[a].push_back(b);
        e[b].push_back(a);
    }
    dfs(1);
    return 0;
}