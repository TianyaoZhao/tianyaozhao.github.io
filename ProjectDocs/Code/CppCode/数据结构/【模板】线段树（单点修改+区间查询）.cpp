// https://www.acwing.com/problem/content/1266/
// 动态求区间和
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
# define lc p << 1
# define rc p << 1 | 1
using namespace std;
typedef pair <int, int> PII;
typedef long long LL;
const int N = 1e5 + 10;
int n, w[N];
struct node{
    int l, r, sum;
}tr[4 * N];

// 建树
void build(int p, int l, int r){ // build(1, 1, n)
    tr[p] = {l, r, w[l]};
    // 是叶子返回
    if(l == r) return;
    // 不是叶子裂开
    int m = l + r >> 1;
    // 递归建树
    build(lc, l, m);
    build(rc, m + 1, r);
    // 回溯时，回传左右儿子的区间和
    tr[p].sum = tr[lc].sum + tr[rc].sum;
}

// 单点修改
void update(int p, int x, int k){ // update(1, 2, k)
    // 是叶子节点修改
    if(tr[p].l == x && tr[p].r == x){
        tr[p].sum += k;
        return;
    }
    // 不是叶子节点裂开
    int m = tr[p].l + tr[p].r >> 1;
    if(x <= m) update(lc, x, k); // 在左儿子
    if(x > m)  update(rc, x, k); // 在右儿子

    // 回溯回传区间和
    tr[p].sum = tr[lc].sum + tr[rc].sum;
}

// 区间查询
int query(int p, int x, int y){ //query(1, x, y)
    // 覆盖则返回
    if(x <= tr[p].l && y >= tr[p].r) return tr[p].sum;
    // 不覆盖则裂开
    int m = tr[p].l + tr[p].r >> 1;
    int sum = 0;
    if(x <= m) sum += query(lc, x, y);
    if(y > m)  sum += query(rc, x, y);
    return sum;
}

void solve(){
    int m;
    cin >> n >> m;
    for(int i = 1; i <= n; i ++) cin >> w[i];
    // 建树
    build(1, 1, n);

    int k, a, b;
    while(m --){
        cin >> k >> a >> b;
        if(k == 0) cout << query(1, a, b) << endl;
        else update(1, a, b);
    }
}
int main(){
    io_speed
    solve();
    return 0;
}