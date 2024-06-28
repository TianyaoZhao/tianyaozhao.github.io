// https://www.acwing.com/problem/content/3726/
// 统计字母出现的次数
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
const int N = 50000 + 10;
char str[N];
int s[26][N]; //26个前缀和
int Q; 

void solve(){
    cin >> str + 1 >> Q;
    // 枚举26个字母
    for(int i = 0; i < 26; i ++){
        // 枚举字符串
        for(int j = 1; str[j]; j ++){
            if(str[j] - 'a' == i){ // 对应字母+1
                s[i][j] = s[i][j - 1] + 1;
            }
            else s[i][j] = s[i][j - 1];
        } 
    }
    int a, b, c, d;
    while(Q --){
       bool same = true;
       cin >> a >> b >> c >> d;  
       // 判断相等不
       for(int i = 0; i < 26; i ++){
            if(s[i][b] - s[i][a - 1] != s[i][d] - s[i][c - 1]){
                same = false;
                break;
            }
       }
       if(same) cout << "DA" << endl;
       else cout << "NE" << endl;
    }
}
int main(){
    io_speed
    solve();
    return 0;
}