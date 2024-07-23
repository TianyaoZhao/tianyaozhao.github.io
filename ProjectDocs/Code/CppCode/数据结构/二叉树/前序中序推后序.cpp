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
string pre, in; // 前序+中序

void dfs(string pre, string in){
    if(pre.empty()) return;
    // 前序遍历的根节点
    char root = pre[0];
    // 中序遍历中根节点的位置
    int k = in.find(root); 
    // 前序：根 左子树 右子树 
    // 中序: 左子树 根 右子树
    
    // 递归左子树
    dfs(pre.substr(1, k) ,in.substr(0, k));
    // 递归右子树
    dfs(pre.substr(k + 1), in.substr(k + 1));
    cout << root;

}

void solve(){
    while(cin >> pre >> in){
        dfs(pre, in);
        cout << endl;
    }
}
int main(){
    io_speed
    solve();
    return 0;
}