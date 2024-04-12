// https://www.acwing.com/problem/content/852/
/* 给出n个点m条边的有向图,可能存在重边和自环，边权非负,求出s号点到n号点的最短距离 */
# include <iostream>
# include <cstring>
# include <algorithm>
# include <queue>
using namespace std;
const int N = 150000 + 10;

struct edge{
    int v, w;
};
vector <edge> e[N];
priority_queue < pair<int, int> > q;     // 大根堆记录当前的点离源点的距离
int dist[N];                             // 记录距离
bool vis[N];                             // 标记是否在集合中
int n, m;
int pre[N];                              // 记录路径
void dijkstra(int s){
    // 初始化所有点到源点的距离为无穷大
    memset(dist, 0x3f, sizeof dist);
    // 源点入队
    dist[s] = -0;
    q.push({dist[s], s});

    while(q.size()){
        auto t = q.top();
        q.pop();
        
        // 取出不在集合中且离源点最近的点
        int u = t.second;
        if(vis[u]) continue; // 再次出队跳过

        vis[u] = true;

        // 枚举所有出边，进行松弛操作
        for(auto ed : e[u]){
            int v = ed.v, w = ed.w;
            if(dist[u] + w < dist[v]){
                dist[v] = dist[u] + w;
                q.push({-dist[v], v});
                pre[v] = u; // 记录前驱节点
            }
        }
    }
}
// 记录前驱节点,反向打印路径
void dfs_path(int u){
    if(u == 1){
        cout << u << " ";
        return;
    }  
    dfs_path(pre[u]);
    cout << u << " ";

}
int main(){
    cin >> n >> m;
    for(int i = 0; i < m; i ++){
        int a, b, c;
        cin >> a >> b >> c;
        e[a].push_back({b, c});
    }
    dijkstra(1);
    
    if(dist[n] == 0x3f3f3f3f)  cout << -1 << endl;
    else cout << dist[n] << endl;

    dfs_path(n);
    return 0;
}