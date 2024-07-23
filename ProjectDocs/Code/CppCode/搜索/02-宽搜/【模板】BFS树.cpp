// queue + 判重数组vis
# include <iostream>
# include <cstring>
# include <algorithm>
# include <queue>
using namespace std;
const int N = 300 + 10;
vector <int> e[N]; 
queue <int> q;
bool vis[N];
int n, m;
void bfs(int u){
    q.push(u);
    vis[u] = true;
    
    while(q.size()){
        int u = q.front();
        q.pop();
        cout << u << "出队" << endl;

        for(auto v: e[u]){
            if(vis[v]) continue;
            q.push(v);
            vis[v] = true;
            cout << v << "入队" << endl;
        }
    }
}
int main(){
    cin >> n >> m;
    for(int i = 0; i < m; i ++){
        int a, b;
        cin >> a >> b;
        e[a].push_back(b);
        e[b].push_back(a);
    }
    bfs(1);
    return 0;
}