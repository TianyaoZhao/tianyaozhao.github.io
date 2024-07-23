// https://www.acwing.com/problem/content/800/
/* 给定一个二维数组，对区域的加减操作转化为对差分二维数组的加减操作 */
# include <iostream>
# include <cstring>
# include <algorithm>
using namespace std;
const int N = 1000 + 10;
int n, m, q, a[N][N], b[N][N];
int main(){
    cin >> n >> m >> q;
    for(int i = 1; i <= n; i ++)
        for(int j = 1; j <= m; j ++) 
            cin >> a[i][j];
    // 二维差分
    for(int i = 1; i <= n; i ++)
        for(int j = 1; j <= m; j ++)
            b[i][j] = a[i][j] - a[i - 1][j] - a[i][j - 1] + a[i - 1][j - 1];
    // 区间加减
    while(q --){
        int x1, x2, y1, y2, c;
        cin >> x1 >> y1 >> x2 >> y2 >> c;
        b[x1][y1] += c;
        b[x1][y2 + 1] -= c;
        b[x2 + 1][y1] -= c;
        b[x2 + 1][y2 + 1] += c;
    }
    // 二维前缀和
    for(int i = 1; i <= n; i ++){
        for(int j = 1; j <= m; j ++){
            a[i][j] = a[i - 1][j] + a[i][j - 1] - a[i - 1][j - 1] + b[i][j];
            cout << a[i][j] << " ";
        }
        cout << endl;
    }
        
            
    return 0;
}