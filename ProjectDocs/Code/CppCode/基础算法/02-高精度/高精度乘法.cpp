// https://www.acwing.com/problem/content/795/
# include <iostream>
# include <cstring>
# include <algorithm>
# include <string>
using namespace std;
const int N = 1e5 + 10;
int a[N], b[N], c[N];
int la, lb, lc;
string sa, sb;
// 高精度乘法:两个正数相乘, 下标之和相等的在c数组的同一个位置
void mul(int a[], int b[]){
    for(int i = 0; i < la; i ++){
        for(int j = 0; j < lb; j ++){
            c[i + j] += a[i] * b[j];
        }
    }
    for(int i = 0; i < lc; i ++){
        c[i + 1] += c[i] / 10;    // 进位
        c[i] %= 10;               // 存余
    }
    // 处理前导0
    while(lc > 0 && c[lc] == 0) lc --;
}
int main(){
    cin >> sa >> sb;
    la = sa.size();
    lb = sb.size();
    lc = la + lb;
    for(int i = 0; i < la; i ++) a[i] = sa[la - 1 - i] - '0';
    for(int i = 0; i < lb; i ++) b[i] = sb[lb - 1 - i] - '0';
    mul(a, b);
    for(int i = lc; i >= 0; i --) cout << c[i];
    return 0;
}