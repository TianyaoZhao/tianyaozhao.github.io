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
int a[N], cnt, n;
// up:从u节点开始上浮
void up(int u){
    // 当前节点有父亲节点，且比父亲节点小，交换，上浮u/2
    if(u / 2 && a[u] < a[u / 2]){
        swap(a[u], a[u / 2]);
        up(u / 2);
    }
}
// push:向堆中插入元素
void push(int x){
    // 插入堆尾, 上浮调整
    a[++ cnt] = x;
    up(cnt);
}
// down:从u节点开始下沉
void down(int u){
    // 寻找u,2u,2u+1最小的节点来进行交换, 交换完之后down
    int v = u;
    if(2 * u <= cnt && a[2 * u] < a[v]) v = 2 * u;
    if(2 * u + 1 <= cnt && a[2 * u + 1] < a[v]) v = 2 * u + 1;
    if(v != u) swap(a[v], a[u]), down(v);
}
// pop:弹出堆顶元素
void pop(){
    cout << a[1] << " ";
    // 堆尾元素赋值给堆顶, down;
    a[1] = a[cnt --];
    down(1);
}
// 删除任意节点
void del(int k){
    a[k] = a[cnt --];
    // 删除之后新加入的元素是大了还是小了，都操作一遍, 只能执行一个
    up(k);
    down(k);
}
// 修改任意节点
void modify(int k, int x){
    a[k] = x;
    up(k);
    down(k);
}
void solve(){
    cin >> n;
    // 构造堆插入
    for(int i = 0; i < n; i ++){
        int x; cin >> x;
        push(x);
    }
    // 堆排序输出堆顶元素
    for(int i = 0; i < n; i ++){
        pop();
    }
}
int main(){
    io_speed
    solve();
    return 0;
}