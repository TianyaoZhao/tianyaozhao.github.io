// https://www.acwing.com/problem/content/1170/
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
const int N = 1e3 + 10;
struct edge{
    int v, w;
};
vector <edge> e[N];
int dist[N];
queue<int> q;
bool vis[N];
int cnt[N]; // 判断负环
int n, m, s;
bool spfa(int s){
    memset(dist, 0x3f, sizeof dist);
    dist[s] = 0;
    q.push(s);
    vis[s] = true;

    while(q.size()){
        int u = q.front();
        q.pop(); 
        vis[u] = false;

        for(auto ed : e[u]){
            int v = ed.v, w = ed.w;
            if(dist[v] > dist[u] + w){
                cnt[v] = cnt[u] + 1;
                dist[v] = dist[u] + w;
                if(!vis[v]){
                    q.push(v);
                    vis[v] = true;
                }
            }
            // n个节点 n - 1 条边，如果等于 n-1 必然有负权回路
            if(cnt[v] > n - 1) return true;
        }
    }
    return false;
}
void solve(){
    cin >> n >> m >> s;
    int a, b, c;
    for(int i = 0; i < m; i ++){
        cin >> a >> b >> c;
        e[a].push_back({b, c});
    }
    bool flag = spfa(s);
    if(flag) cout << -1;
    if(!flag){
        for(int i = 1; i <= n; i ++){
            if(i == s) cout << 0 << endl;
            else{
                 if(dist[i] < 1e6 + 10) cout << dist[i] << endl;
                 else cout << "NoPath" << endl;
            }
        }
    }
}
int main(){
    io_speed
    solve();
    return 0;
}