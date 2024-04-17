// https://www.luogu.com.cn/problem/P1055
# include <iostream>
# include <cstring>
# include <algorithm>
# include <string>
using namespace std;
const int N = 100;
char s[N];
int a[N], k;
int main(){
    cin >> s + 1;
    for(int i = 1; i <= strlen(s + 1);  i ++){
        if(s[i] == '-') continue;
        a[++ k] = s[i] - '0';
    }

    int res = 0;
    for(int i = 1; i <= k - 1; i ++){
        res += a[i] * i;
    }

    res = res % 11;

    if(res == 10 && a[k] == 'X' - '0') cout << "Right";
    else if(res == 10 && a[k] != 'X' - '0'){
        for(int i = 1; i < strlen(s + 1); i ++){
            cout << s[i];
        }
        cout << 'X';
    }
    else if(res == a[k]) cout << "Right";
    else{
        for(int i = 1; i < strlen(s + 1); i ++){
            cout << s[i];
        }
        cout << res;
    }
    return 0;
}