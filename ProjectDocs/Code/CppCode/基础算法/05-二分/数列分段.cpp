// https://www.acwing.com/problem/content/4181/
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
int n, m;
int a[N];
bool check(LL mid){

}
void solve(){
    // 二分每段和的最大值, check当前局面是否能找到比mid还小的和
    cin >> n >> m;
    LL sum = 0;
    for(int i = 1; i <= n; i ++){
        cin >> a[i];
        sum += a[i];
    }
    LL l = 0, r = sum + 1;
    while(l + 1 < r){
        LL mid = l + r >> 1;
        if(check(mid)) r = mid;
        else l = mid;
    }
    cout << r;


}
int main(){
    io_speed
    solve();
    return 0;
}