// https://www.luogu.com.cn/problem/P1012
// 数组中数字首尾相接，组成一个最大的整数
# include <iostream>
# include <cstring>
# include <algorithm>
# include <string>
using namespace std;
int n;
const int N = 20 + 10;
string s[N];
// const int N = 
// 写一个cmp函数，排序的规则是从最高位开始排，谁大谁在前
bool cmp(string a, string b){
    return a + b > b + a;
}
int main(){
    cin >> n;
    for(int i = 0; i < n; i ++) cin >> s[i];
    sort(s, s + n, cmp);
    for(int i = 0; i < n; i ++) cout << s[i];
}