// https://www.acwing.com/problem/content/799/
/* 给定一段序列，对序列某个区间进行加减操作，转化为对差分序列的加减操作  */
# include <iostream>
# include <cstring>
# include <algorithm>
using namespace std;
const int N = 1e5 + 10;
int n, m, a[N], b[N];
int main(){
    cin >> n >> m;
    for(int i = 1; i <= n; i ++) cin >> a[i];
    // 差分
    for(int i = 1; i <= n; i ++) b[i] = a[i] - a[i - 1];

    while(m --){
        int l, r, c;
        cin >> l >> r >> c;
        b[l] += c;
        b[r + 1] -= c;
    }

    // 前缀和
    for(int i = 1; i <= n; i ++){
        a[i] = a[i - 1] + b[i];
        cout << a[i] << " ";
    }

    return 0;
}