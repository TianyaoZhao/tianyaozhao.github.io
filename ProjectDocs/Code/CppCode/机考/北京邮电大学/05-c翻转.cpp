// https://www.acwing.com/problem/content/3538/
// 矩阵翻转
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
const int N = 5 + 10, n = 5;
int m[N][N], backup[N][N];
void roate(int a, int b, int x, int y){
    memcpy(backup, m, sizeof m);
    if(a == 1){ // 顺时针
        for(int i = x; i < x + b; i ++){
            for(int j = y; j < y + b; j ++){
                backup[j + x - y][x + y + b - i - 1] = m[i][j];
            }
        }
    }
    else{ // 逆时针
        for(int i = x; i < x + b; i ++){
            for(int j = y; j < y + b; j ++){
                backup[x + y + b - j - 1][i + y - x] = m[i][j];
            }
        }
    }
    memcpy(m, backup, sizeof m);
    

}
void solve(){
    for(int i = 1; i <= n; i ++)
        for(int j = 1; j <= n; j ++)
            cin >> m[i][j];

    int a, b, x, y;
    cin >> a >> b >> x >> y;

    roate(a, b, x, y);
    
    for(int i = 1; i <= n; i ++){
        for(int j = 1; j <= n; j ++){
            cout << m[i][j] << " ";
        }
        cout << endl;
    }
}
int main(){
    io_speed
    solve();
    return 0;
}