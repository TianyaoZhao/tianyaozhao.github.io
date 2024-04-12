// https://www.acwing.com/problem/content/844/
/* 给定数字1~n,按照字典序，输出数字的全排列 */
# include <iostream>
# include <cstring>
# include <algorithm>
using namespace std;
const int N = 10 + 10;
int path[N];
bool vis[N];
int n;

void dfs(int u){
    if(u == n + 1){ // 搜到第n个位置，得到当前的答案
        for(int i = 1; i <= n; i ++) cout << path[i] << " ";
        cout << endl;
    }   
    for(int i = 1; i <= n; i ++){
        if(vis[i]) continue;
        vis[i] = true;
        path[u] = i;  // 下
        
        dfs(u + 1);
                
        vis[i] = false;
        path[u] = 0;  // 回 恢复现场
        
    }
}
int main(){
    cin >> n;
    dfs(1);  // 一共有n个位置,从第一个位置开始搜
    return 0;
}
