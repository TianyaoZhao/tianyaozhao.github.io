// https://www.acwing.com/problem/content/1499/
# include <iostream>
# include <cstring>
# include <algorithm>
# include <string>
# include <cmath>
# include <stack>
# include <queue>
# include <vector>
# include <iomanip>
# include <unordered_map> 
# define io_speed ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
# define endl '\n'
using namespace std;
typedef pair <int, int> PII;
typedef long long LL;
const int N = 30 + 10;
int postorder[N], inorder[N]; //  后序遍历和中序遍历
unordered_map <int, int> l, r, pos; // 哈希表，存左儿子，右儿子以及每个节点在中序遍历中的下标
int n;
queue <int> q;
// 中序遍历左端点，中序遍历右端点
// 后序遍历左端点，后序遍历右端点
int build(int il, int ir, int pl, int pr){
    // 根节点为后序遍历的右端点
    int root = postorder[pr];
    int k = pos[root]; // 中序遍历中根节点的下标
    // 左子树存在 递归建立左子树
    // k - 1 - il = x - pl 
    // 中序: 左子树 根 右子树
    // 后序：左子树 右子树 根 找对应关系建树
    if(il < k) l[root] = build(il, k - 1, pl, pl + k - 1 - il);
    // 右子树存在递归建立右子树
    if(ir > k) r[root] = build(k + 1, ir, pl + k - 1 - il + 1, pr - 1);
    return root;
}

void bfs(int root){
    q.push(root);
    while(q.size()){
        auto t = q.front();
        q.pop();
        cout << t << " ";
        // 有左儿子
        if(l.count(t)) q.push(l[t]);
        // 有右儿子
        if(r.count(t)) q.push(r[t]);
    }
}
void solve(){
    cin >> n; 
    for(int i = 0; i < n; i ++) cin >> postorder[i];
    for(int i = 0; i < n; i ++){
        cin >> inorder[i];
        pos[inorder[i]] = i;  // 存下标
    }
    // 建树，返回根节点
    int root = build(0, n - 1, 0, n - 1);
    bfs(root);
}
int main(){
    io_speed
    solve();
    return 0;
}