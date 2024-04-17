// https://www.luogu.com.cn/problem/P5015
# include <iostream>
# include <cstring>
# include <algorithm>
# include <string>
using namespace std;
string s;
int main(){
    getline(cin, s);
    int cnt = 0;
    for(int i = 0; i < s.size(); i ++){
        if(s[i] == ' ' || s[i] == '\n') continue;
        cnt ++;
    }
    cout << cnt;
    return 0;
}