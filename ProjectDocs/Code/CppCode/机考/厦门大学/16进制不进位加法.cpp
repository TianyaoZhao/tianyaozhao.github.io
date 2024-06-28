// https://www.acwing.com/problem/content/3705/
// 16进制不进位加法,比如 A + 6 = 16(十进制) = 进1位余数0,所以A + 6 = 0
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
const int N =  100 + 10;
string sa, sb;
int a[N], b[N];
char c[N];
int la, lb, lc;
void add(int a[], int b[], char c[]){
    for(int i = 0; i < lc; i ++){
        int t = (a[i] + b[i]) % 16;
        if(t >= 0 && t <= 9) c[i] = t + '0';
        else c[i] = t - 10 + 'A';
    }
    for(int i = lc; i >= 0; i --) cout << c[i];
    cout << endl;
}
void solve(){
    while(cin >> sa >> sb){
        la = sa.size(), lb = sb.size();
        lc = max(la, lb);
        for(int i = 0; i < la; i ++){
            if(sa[la - i - 1] >= 'A' && sa[la - i - 1] <= 'F') a[i] = sa[la - i - 1] - 'A' + 10;
            else a[i] = sa[la - i - 1] - '0';
        } 
        for(int i = 0; i < lb; i ++){
            if(sb[lb - i - 1] >= 'A' && sb[lb - i - 1] <= 'F') b[i] = sb[lb - i - 1] - 'A' + 10;
            else b[i] = sb[lb - i - 1] - '0';
        }
        add(a, b, c);

        memset(a, 0, sizeof a);
        memset(b, 0, sizeof b);
        memset(c, ' ', sizeof c);
    }

}
int main(){
    io_speed
    solve();
    return 0;
}