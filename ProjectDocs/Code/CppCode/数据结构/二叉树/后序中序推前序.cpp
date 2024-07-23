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
string post, in; // 后序+中序

void dfs(string post, string in){
    // cout << post.size() << endl;
    if(post.empty()) return;
    // 后序遍历的根节点
    char root = post[post.size() - 1];
    // 中序遍历中根节点的位置
    int k = in.find(root); 
    // 后序: 左子树 右子树 根
    // 中序: 左子树 根 右子树
    
    cout << root;
    // 递归左子树
    dfs(post.substr(0, k) ,in.substr(0, k));
    // 递归右子树
    dfs(post.substr(k, post.size() - k -1), in.substr(k + 1));
}

void solve(){
    while(cin >> post >> in){
        dfs(post, in);
        cout << endl;
    }
}
int main(){
    io_speed
    solve();
    return 0;
}