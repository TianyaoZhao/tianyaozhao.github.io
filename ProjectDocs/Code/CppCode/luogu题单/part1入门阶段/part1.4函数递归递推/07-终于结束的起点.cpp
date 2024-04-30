    // https://www.luogu.com.cn/problem/P4994
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
    const int N = 1e7 + 10;
    LL f[N];
    LL m, cnt = 1;
    void solve(){
        cin >> m;
        f[0] = 0, f[1] = 1;
        for(int i = 2; i < N - 1; i ++){
            f[i] = f[i - 1] + f[i - 2];
        }
        int idx = 2;
        while(true){
            if(f[idx] % m == 1 && f[idx + 1] % m == 0){
                cout << idx;
                break;
            }
            idx ++;
        }
    }
    int main(){
        io_speed
        solve();
        return 0;
    }
