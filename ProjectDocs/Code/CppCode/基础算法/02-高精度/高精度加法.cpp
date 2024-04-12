// https://www.acwing.com/problem/content/description/793/
#include <iostream>
#include <cstring>
#include <algorithm>
#include <string>
using namespace std;
const int N = 1e5 + 10;
int a[N], b[N], c[N];
string sa, sb;
int la, lb, lc;

// 高精度加法：适用于两个正数相加
void add(int a[], int b[]){
    for (int i = 0; i < lc; i++){
        c[i] += a[i] + b[i];   // 累加
        c[i + 1] += c[i] / 10; // 进位
        c[i] %= 10;            // 存余
    }
    if (c[lc]) lc++; // 最高位有进位
}
int main(){
    cin >> sa >> sb;
    la = sa.size();
    lb = sb.size();
    lc = max(la, lb);
    // 反向存入数组,使得低位从下标0开始
    for (int i = 0; i < la; i++) a[i] = sa[la - 1 - i] - '0';
    for (int i = 0; i < lb; i++) b[i] = sb[lb - 1 - i] - '0';
    add(a, b);
    for (int i = lc - 1; i >= 0; i--) cout << c[i];
    return 0;
}
