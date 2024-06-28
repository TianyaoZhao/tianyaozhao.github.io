// https://www.acwing.com/problem/content/854/
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
int cnt[N];    // 记录边数,判负环
queue <int> q; // 维护刚更新的边的，终点
struct edge{
    int v, w;
};
vector <edge> e[N]; // 邻接表存图


bool spfa(int s){
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
                // 出边终点v被更新，且不在队内，就可以进入队列去更新其他的节点
                if(!vis[v]) q.push(v), vis[v] = true;
                // 记录走过的边数
                cnt[v] = cnt[u] + 1;
                // n + 1个节点,n条边，如果大于等于n说明一定有负环
                if(cnt[v] > n) return true;
            }
        }
    }
}
void solve(){
    cin >> n >> m;
    for(int i = 0; i < m; i ++){
        int u, v, w;
        cin >> u >> v >> w;
        e[u].push_back({v, w});
    }
    // 虚拟点，到其他所有点的距离都是0
    // 防止有孤岛，然后孤岛有负环且恰好起点不在这个孤岛上
    for(int i = 1; i <= n; i ++) e[0].push_back({i, 0});
    
    if(spfa(0)) puts("Yes");
    else puts("No");
}
int main(){
    io_speed
    solve();
    return 0;
}
