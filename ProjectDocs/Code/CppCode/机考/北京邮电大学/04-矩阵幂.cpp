// https://www.acwing.com/problem/content/3537/
// 求一个方阵的k次幂
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
const int N = 10 + 5;
int w[N][N], res[N][N];
int n, k;
// 矩阵a * b 结果保存到c中
void mul(int a[][N], int b[][N], int c[][N]){
    int tmp[N][N];
    memset(tmp, 0, sizeof tmp);
    for(int i = 0; i < n; i ++)
        for(int j = 0; j < n; j ++)
            for(int k = 0; k < n; k ++)
                tmp[i][j] += a[i][k] * b[k][j];
    memcpy(c, tmp, sizeof tmp); 
}

void solve(){
    cin >> n >> k;
    // 读入原始矩阵 
    for(int i = 0; i < n; i ++)
        for(int j = 0; j < n; j ++)
            cin >> w[i][j];
    // 将res初始化为单位矩阵 单位矩阵乘5次w
    for(int i = 0; i < n; i ++) res[i][i] = 1;
    while(k --) mul(res, w, res);
    // 输出结果矩阵 
    for(int i = 0; i < n; i ++){
        for(int j = 0; j < n; j ++)
            cout << res[i][j] << " ";
        cout << endl; 
    } 
}
int main(){
    io_speed
    solve();
    return 0;
}