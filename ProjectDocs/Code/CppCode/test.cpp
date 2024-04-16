# include <iostream>
# include <cstring>
# include <algorithm>
# include <vector>
# include <string>
using namespace std;
const int N = 1e3 + 5;
int a[N][N];
int res[N][N];
int n, m, k;
int main(){
    cin >> n >> m >> k;
    for(int i = 1; i <= n; i ++){
        for(int j = 1; j <= m; j ++){
            cin >> a[i][j];
        }
    }

    // 枚举每套题
    for(int j = 1; j <= m; j ++){
        for(int i = 1; i <= n; i ++){
            int t = a[i][j]; // 表示第j套题需要安排在第t天
            res[j][t] ++;    // 表示第j套题，第t天需要安排
        }
    }

    // 枚举每一天
    int cnt = 0;
    for(int i = 1; i <= k; i ++){
        for(int j = 1; j <= m; j ++){
            if(res[j][i] != 0) cnt ++;
        }
        cout << cnt << " ";
        cnt = 0; 
    }
    return 0;
}

