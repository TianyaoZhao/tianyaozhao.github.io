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
const int N = 20 + 5;
LL w[N][N][N];

int func(int a, int b, int c){
    if(a <= 0 || b <= 0 || c <= 0) return 1;
    if(a > 20 || b > 20 || c > 20) return w[20][20][20] = func(20, 20, 20);
    if(w[a][b][c] != 0) return w[a][b][c];
    if(a < b && b < c) return w[a][b][c] = func(a, b, c - 1) + func(a, b - 1, c - 1) - func(a, b - 1, c);
    return w[a][b][c] = func(a - 1, b, c) + func(a - 1, b - 1, c) + func(a - 1, b, c - 1) - func(a - 1, b - 1, c - 1);
}
void solve(){
    LL a, b, c;
    while(cin >> a && cin >> b && cin >> c){
        if(a == -1 && b == - 1 && c == -1) break;
        cout << "w(" << a << ", " << b << ", " << c << ')' << " = " << func(a,b,c) << endl;
    }
}
int main(){
    io_speed
    solve();
    return 0;
}