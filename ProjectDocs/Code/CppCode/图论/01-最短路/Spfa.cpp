// https://www.acwing.com/problem/content/853/
/* 给定一个n个点m条边的有向图，图中可能存在重边和自环，边权可能为负数。*/
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
const int N = 1e5 + 10;
int n, m;
int dist[N];   // 记录距离
int vis[N];    // 标记是否在队列内
int pre[N];    // 记录路径
queue <int> q; // 维护刚更新的边的，终点
struct edge{
    int v, w;
};
vector <edge> e[N]; // 邻接表存图

void spfa(int s){
    // 初始化其他点到源点的距离为无穷,dist[s] = 0
    memset(dist, 0x3f, sizeof dist);
    dist[s] = 0;
    // 源点入队，标记在队列中
    q.push(s);
    vis[s] = true;

    while(q.size()){
        // 取出队头
        int u = q.front();
        q.pop();
        vis[u] = false;

        // 遍历u的邻边进行松弛操作
        for(auto ed : e[u]){
            int v = ed.v, w = ed.w;
            if(dist[v] > dist[u] + w){
                dist[v] = dist[u] + w;
                pre[v] = u;//记录路径
                // 出边终点v被更新，且不在队内，就可以进入队列去更新其他的节点
                if(!vis[v]) q.push(v), vis[v] = true;
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
    spfa(1);

    if(dist[n] < 0x3f3f3f3f / 2) cout << dist[n];
    else cout << "impossible";
}
int main(){
    io_speed
    solve();
    return 0;
}
