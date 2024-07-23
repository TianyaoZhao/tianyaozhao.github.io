// https://www.acwing.com/problem/content/1266/
// 动态求区间和,动态修改区间，sum开longlong
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
struct node{
    int l, r, sum, add;
}tr[4 * N];
int n, w[N];
// 建树 build(1, 1, n)
void build(int p, int l, int r){ 
    tr[p] = {l, r, w[l]};
    // 叶子节点则返回
    if(l == r) return;
    // 非叶子节点则裂开
    int m = l + r >> 1; 
    build(lc, l, m);
    build(rc, m + 1, r);

    // 回溯回传区间和
    tr[p].sum = tr[lc].sum + tr[rc].sum; 
}

// 单点修改 update(1, x, k)
void update(int p, int x, int k){
    // 叶子节点则修改
    if(tr[p].l == x && tr[p].r == x){
        tr[p].sum += k;
        return;
    }
    // 非叶子节点则裂开
    int m = tr[p].l + tr[p].r >> 1;
    if(x <= m) update(lc, x, k);
    if(x > m)  update(rc, x, k);

    // 回溯回传区间和
    tr[p].sum = tr[lc].sum + tr[rc].sum;
}

// 懒标记下传 pushdown(p)
void pushdown(int p){
    if(tr[p].add){ //有懒标记
        // 修改左右儿子的sum
        tr[lc].sum += tr[p].add *(tr[lc].r - tr[lc].l + 1);
        tr[rc].sum += tr[p].add *(tr[rc].r - tr[rc].l + 1);
        // 下传懒标记
        tr[lc].add += tr[p].add;
        tr[rc].add += tr[p].add;
        // 父亲懒标记清空
        tr[p].add = 0;
    }
}
// 区间修改 modify(1, x, y, k)
void modify(int p, int x, int y, int k){
    // 覆盖则懒惰修改
    if(x <= tr[p].l && y >= tr[p].r){
        tr[p].sum += k * (tr[p].r - tr[p].l + 1);
        tr[p].add += k;  // 懒标记
        return;
    }
    // 不覆盖则则裂开
    int m = tr[p].l + tr[p].r >> 1;
    pushdown(p); // 下传懒标记
    if(x <= m) modify(lc, x, y, k);
    if(y > m) modify(rc, x, y, k);

    // 回溯回传区间和
    tr[p].sum = tr[lc].sum + tr[rc].sum;
}

// 区间查询 query(1, x, y)
int query(int p, int x, int y){
    // 覆盖则返回
    if(x <= tr[p].l && y >= tr[p].r) return tr[p].sum;
    // 不覆盖则裂开
    int m = tr[p].l + tr[p].r >> 1;
    pushdown(p); // 下传懒标记
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

    int k, a, b, c; 
    while(m --){
        cin >> k >> a >> b;
        if(k == 2){
            cout << query(1, a, b) << endl;
        }
        else{
            cin >> c;
            modify(1, a, b, c);
        }
    }
}
int main(){
    io_speed
    solve();
    return 0;
}