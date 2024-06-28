// https://www.acwing.com/problem/content/3695/
/* 输入两个字符串 s1,s2 输出最长连续公共子串长度和最长连续公共子串。*/
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
const int N = 100 + 10;
char s1[N], s2[N];
char s[N];        // 存最终结果
int f[N][N];      // 以s1[i], s2[j]结尾的最长公共子串的长度
// 状态计算, s1[i] == s2[j] f[i][j] = f[i - 1][j - 1] + 1 else f[i][j] = 0
void solve(){
    int ans = 0, index = 0;
    cin >> s1 + 1 >> s2 + 1;
    for(int i = 1; i <= strlen(s1 + 1); i ++){
        for(int j = 1; j <= strlen(s2 + 1); j ++){
            if(s1[i] == s2[j]){
                f[i][j] = f[i - 1][j - 1] + 1;
                // 存最后一次ans更新时,s1[i]的下标,更新ans
                if(f[i][j] >= ans){
                    ans = f[i][j];
                    index = i;
                }
            }
            else{
                f[i][j] = 0;
            }
        }
    }

    cout << ans << endl;
    for(int i = index - ans + 1; i <= index; i ++) cout << s1[i];
}
int main(){
    io_speed
    solve();
    return 0;
}