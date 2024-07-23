// https://www.acwing.com/problem/content/3711/
/* 给定一个 n×m的整数矩阵，行的编号为 1∼n，列的编号为 1∼m，求矩阵中的所有鞍点。
鞍点，即该位置上的元素在该行上最大，在该列上最小。*/

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
const int N = 10 + 10;
int a[N][N], n, m;
int rowmax[N], colmin[N];
void solve(){
    cin >> n >> m;
    for(int i = 1; i <= n; i ++){
        for(int j = 1; j <= m; j  ++){
            cin >> a[i][j];
        }
    }
    // 统计行最大值
    for(int i = 1; i <= n; i ++){
        for(int j = 1; j <= m; j ++){
            rowmax[i] = max(rowmax[i], a[i][j]);
        }
    }
    // 统计列最小值
    memset(colmin, 0x3f, sizeof colmin);
    for(int j = 1; j <= m; j ++){
        for(int i= 1; i <= n; i ++){
            colmin[j] = min(colmin[j], a[i][j]);
        }
    }

    // 查鞍点
    bool flag = false;
    for(int i = 1; i <= n; i ++){
        for(int j = 1; j <= m; j ++){
            if(a[i][j] == rowmax[i] && a[i][j] == colmin[j]){
                cout << i << " " << j << " " << a[i][j] << endl;
                flag = true;
            }
        }
    }
    if(!flag) cout << "NO";
}
int main(){
    io_speed
    solve();
    return 0;
}