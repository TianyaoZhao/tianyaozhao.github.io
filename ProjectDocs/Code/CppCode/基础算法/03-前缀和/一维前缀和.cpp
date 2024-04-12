// https://www.acwing.com/problem/content/797/
/* 给定一组序列，求某一段区间的和是多少 */
# include <iostream>
# include <cstring>
# include <algorithm>
using namespace std;
const int N = 1e5 + 10;
int n, m, a[N], s[N];
int main(){
    cin >> n >> m;
    for(int i = 1; i <= n; i ++) cin >> a[i];
    
    for(int i = 1; i <= n; i ++) s[i] = s[i - 1] + a[i];

    while(m --){
        int l, r;
        cin >> l >> r;
        cout << s[r] - s[l - 1] << endl;
    }
    return 0;
}