// https://www.acwing.com/problem/content/852/
/* 给出n个点m条边的有向图,可能存在重边和自环，边权非负,求出s号点到n号点的最短距离 */
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
const int N = 150000 + 10;
int dist[N]; // 记录距离
int vis[N];  // 记录在不在集合内
int pre[N];    // 记录路径

priority_queue <PII, vector<PII>, greater<PII>> q; // 用小根堆维护每次更新距离后离源点最近的点
struct edge{
    int v, w;
};
vector<edge> e[N];  // 邻接表写法
int n, m;

void dijkstra(int s){
    // 初始化所有点到源点的距离为0x3f3f3f3f,dist[s] = 0
    memset(dist, 0x3f, sizeof dist);
    dist[s] = 0;

    // 源点入小根堆
    q.push({dist[s], s});

    while(q.size()){
        // 选出不在集合中且离源点最近的点
        auto t = q.top();
        q.pop();

        // 取出队头节点编号
        int u  = t.second;
        // 再出队跳过,没有就加入到集合中
        // 被两个参考点更新，同时入队
        if(vis[u]) continue;
        vis[u] = true; 

        // 遍历队头节点的所有邻边进行松弛操作
        for(auto ed : e[u]){
            int v = ed.v, w = ed.w;
            if(dist[v] > dist[u] + w){
                dist[v] = dist[u] + w;
                q.push({dist[v], v});
                pre[v] = u;   //记录路径
            }
        }  
    }

}
void dfs_path(int u){
    if(u == 1){
        cout << u << " ";
        return;
    }
    dfs_path(pre[u]);
    cout << u << " ";
}
void solve(){
    cin >> n >> m;
    for(int i = 0; i < m; i ++){
        int u, v, w;
        cin >> u >> v >> w;
        e[u].push_back({v, w});
    }
    dijkstra(1);

    if(dist[n] == 0x3f3f3f3f) cout << -1;
    else cout << dist[n];
}
int main(){
    io_speed
    solve();
    return 0;
}