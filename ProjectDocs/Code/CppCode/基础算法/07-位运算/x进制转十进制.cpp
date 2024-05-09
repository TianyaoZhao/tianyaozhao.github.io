// 输入x进制数，输出10进制数
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
const int N = 100 + 5;
char s[N]; // x进制数  
int x;     // x进制
void x2ten(char s[], int x){
    int ans = 0;// 结果
    for(int i = 0; i < strlen(s); i ++){
        ans = ans * x;
        if(s[i] >= '0' && s[i] <= '9') ans += s[i] - '0';
        else ans += s[i] - 'A' + 10;
    }
    cout << ans;
}
void solve(){
    cin >> s >> x;
    x2ten(s, x);
}
int main(){
    io_speed
    solve();
    return 0;
}