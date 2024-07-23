// https://www.acwing.com/problem/content/3387/
// 给定先序遍历的字符串。建立二叉树输出中序遍历
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

const int N = 1e8 + 10;
/* 这两个数组声明顺序不能有问题 */
char tree[N];// 数组模拟二叉树
char str[N]; // 输入的字符串
int length;//字符串长度
int cnt;//当前树内节点个数


// 根据先序遍历建树
void build(int u){
    cnt ++;
    if(cnt > length) return;
    if(str[cnt] == '#') return;
    
    tree[u] = str[cnt];    // 根节点
    build(2 * u);      // 左子树
    build(2 * u + 1);  // 右子树
}


void dfs(int u){
    if(!tree[u]) return;
    dfs(2 * u);
    cout << tree[u] << " ";
    dfs(2 * u + 1);
}


void solve(){
    cin >> str + 1;
    length = strlen(str + 1);
    build(1); // 建树
    dfs(1);   // 中序遍历
}
int main(){
    io_speed
    solve();
    return 0;
}

