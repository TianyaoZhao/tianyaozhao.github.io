// https://www.acwing.com/problem/content/3722/
// 添加边,使得图为无向连通图,任意两个点之间之间可达
// 并查集,把连通的节点放入集合中,查祖宗节点,然后对于没有连通的就是需要加边的
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
const int N = 1000 + 10;
int n, m;
int fa[N];
int find(int x){
    if(x == fa[x]) return x;
    else return fa[x] = find(fa[x]);
}
void solve(){
    cin >> n >> m;
    for(int i = 1; i <= n; i ++) fa[i] = i;
    int x, y;
    while(m --){
        cin >> x >> y;
        fa[find(x)] = find(y); // x的祖宗节点修改为y的祖宗节点
    }

    // 查询所有祖宗节点,去重即可
    vector <int> res;
    for(int i = 1; i <= n; i ++) res.push_back(find(i));
    // 排序 + 去重
    sort(res.begin(), res.end());
    res.erase(unique(res.begin(), res.end()), res.end());
    cout << res.size() - 1;


}
int main(){
    io_speed
    solve();
    return 0;
}