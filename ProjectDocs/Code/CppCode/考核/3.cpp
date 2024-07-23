// https://leetcode.cn/problems/reverse-words-in-a-string/description/
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
const int N = 1e4 + 10;
int idx[N];
string s[N];
string str;
void solve(){
    getline(cin, str);
    // cout << str;
    string t;
    int k = 0;
    bool flag = false;
    for(int i = 0; i < str.size(); i ++){
        if(str[i] != ' '){
            t += str[i];
            flag = false;
        }
        else{
            if(!flag){
                s[k ++] = t;
                t = "";
                flag = true;
            }

        }
    }
    s[k ++] = t;
    for(int i = k - 1; i >= 0 ; i --){
        if(s[i] == " ") continue;
        else{
            if(i == 0) cout << s[i];
            else cout << s[i] << " ";
        }
    }
}
int main(){
    io_speed
    solve();
    return 0;
}