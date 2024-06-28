// https://www.nowcoder.com/practice/d6bd75dbb36e410995f8673a6a2e2229?tpId=63&tqId=29595&tPage=2&ru=/kaoyan/retest/9001&qru=/ta/zju-kaoyan/question-ranking
// 思路1：排序方式变为st=1的优先级高, 排在最前面
// 四路2：如果检测到一个道路已经建好，就将他的花费置为0， 然后用Kruskal
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
const int N = 100 * 100 + 10;
struct edge{
    int u, v, w, st;
};
int fa[N];
int find(int x){
    if(fa[x] == x) return x;
    else return fa[x] = find(fa[x]);
}
bool cmp(edge a, edge b){
    if(a.st == 1 && b.st != 1) return true;
    else if(a.st != 1 && b.st == 1) return false;
    else return a.w < b.w;
}
void solve(){
    int n;
    while(cin >> n){
        if(n == 0) break;
        int m = n * (n - 1) / 2;
        for(int i = 1; i <= n; i ++) fa[i] = i;
        int u, v, w, st;
        int res = 0;
        edge e[N];
        for(int i = 0; i < m; i ++){
            cin >> u >> v >> w >> st;
            e[i] = {u, v, w, st};
        }
        sort(e, e + m, cmp);

        for(int i = 0; i < m; i ++){
            int u = e[i].u, v = e[i].v, w = e[i].w, st = e[i].st;
            if(find(u) != find(v)){
                fa[find(u)] = find(v);
                if(st == 0) res += w;
            }
        }
        cout << res << endl;
    }
}
int main(){
    io_speed
    solve();
    return 0;
}