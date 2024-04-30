// https://www.luogu.com.cn/problem/P1068
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
const int N = 5000 + 10;
PII a[N];
int n, m;
bool cmp(PII a, PII b){
    if(a.second == b.second){
        return a.first < b.first;
    }
    else return a.second > b.second;
}
void solve(){
    cin >> n >> m;
    for(int i = 0; i < n; i ++){
        int k, s;
        cin >> k >> s;
        a[i].first = k;
        a[i].second = s;
    }
    sort(a, a + n, cmp);
    int line = a[m * 3 / 2 - 1].second;
    int cnt = 0;
    for(int i = 0; i < n; i ++){
        if(a[i].second >= line) cnt ++;
    }
    cout << line << " ";
    cout << cnt << endl;
    for(int i = 0; i < cnt; i ++){
        cout << a[i].first << " " << a[i].second << endl;
    }
}
int main(){
    io_speed
    solve();
    return 0;
}