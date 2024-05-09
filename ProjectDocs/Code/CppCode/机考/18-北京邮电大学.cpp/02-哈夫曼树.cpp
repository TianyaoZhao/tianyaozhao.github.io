// https://www.acwing.com/activity/content/problem/content/8627/
// 给定n个叶子节点和n个权重，构造哈夫曼树，输出该树带权路径长度
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
// const int N = 
int n;
priority_queue <int, vector<int>, greater <int>> heap; // 小根堆
void solve(){
    cin >> n;
    int x;
    while(n --){
        cin >> x;
        heap.push(x);
    }
    int ans = 0;
    while(heap.size() > 1){ // 剩下一个就不需要合并了
        int x = heap.top();
        heap.pop();
        int y = heap.top();
        heap.pop();
        int sum = x + y;
        ans += sum;         // 每次合并把权重加到结果
        heap.push(sum);     // 合并后的树加到heap中
    }
    cout << ans;
}
int main(){
    io_speed
    solve();
    return 0;
}