// https://www.luogu.com.cn/problem/P1059
// 去重排序
# include <iostream>
# include <cstring>
# include <algorithm>
# include <string>
# include <cmath>
# include <stack>
# include <queue>
# include <vector>
# define io_speed ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
# define endl '\n'
using namespace std;
typedef pair <int, int> PII;
typedef long long LL;
// const int N = 
vector <int> a;
int n;
void solve(){
    cin >> n;
    for(int i = 0; i < n; i ++){
        int x; cin >> x;
        a.push_back(x);
    }
    sort(a.begin(), a.end());
    a.erase(unique(a.begin(), a.end()), a.end());
    cout << a.size() << endl;
    for(auto item : a){
        cout << item << " ";
    }
}
int main(){
    io_speed
    solve();
    return 0;
}