// https://www.acwing.com/problem/content/861/
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
const int N = 200000 + 10;
struct edge{
    int u, v, w;
}e[N];
bool cmp(edge a, edge b){
    return a.w < b.w;
}
int n, m;
int fa[N]; // 并查集存每个节点的祖宗节点
int cnt;   // 记录加入到并查集中的边数
int s;     // 权重总和
int find(int x){
    // 找到祖宗节点
    if(x == fa[x]) return x;
    // 递归查找他的父亲节点的祖宗节点
    else return fa[x] = find(fa[x]);
}
void kruskal(){
    sort(e, e + m, cmp);
    for(int i = 0; i < m; i ++){
        int u = e[i].u, v = e[i].v, w = e[i].w;
        if(find(u) != find(v)){
            fa[find(u)] = find(v); // 把u的祖宗节点值修改为v的祖宗节点
            s += w; cnt ++;
        }   

    }
}
void solve(){
    cin >> n >> m;
    // 初始化祖宗节点为自己本身
    for(int i = 1; i <= n; i ++) fa[i] = i;
    for(int i = 0; i < m; i ++){
        int u, v, w;
        cin >> u >> v >> w;
        e[i] = {u, v, w};
    }
    kruskal();
    if(cnt == n - 1) cout << s;
    else cout << "impossible";
}
int main(){
    io_speed
    solve();
    return 0;
}