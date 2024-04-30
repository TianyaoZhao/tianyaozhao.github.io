// https://www.luogu.com.cn/problem/P1309
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
const int N = 2e5 + 10;
struct person{
    int id;
    int s;
    int w;
}a[N]; 
int n, r, q;
bool cmp(person &a, person &b){
    if(a.s == b.s) return a.id < b.id;
    else return a.s > b.s;
}
void solve(){
    cin >> n >> r >> q;
    for(int i = 0; i < 2 * n; i ++) {
        a[i].id = i + 1;
        cin >> a[i].s;
    }
    for(int i = 0; i < 2 * n; i ++) cin >> a[i].w;
    for(int i = 0; i < r; i ++){
        sort(a, a + 2 * n, cmp);
        for(int i = 0; i < 2 * n; i += 2){
            if(a[i].w > a[i + 1].w){
                a[i].s += 1;
            } 
            else a[i + 1].s += 1;
        }
    }
    sort(a, a + 2 * n, cmp);
    cout << a[q - 1].id;
}
int main(){
    io_speed
    solve();
    return 0;
}