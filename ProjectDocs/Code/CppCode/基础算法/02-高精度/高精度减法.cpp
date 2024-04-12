// https://www.acwing.com/problem/content/794/
#include <iostream>
#include <cstring>
#include <algorithm>
#include <string>
using namespace std;
const int N = 1e5 + 10;
int a[N], b[N], c[N];
int la, lb, lc;
string sa, sb;

// 比较a和b的大小
bool cmp(int a[], int b[]){
    // 位数不同
    if (la != lb) return la > lb;
    // 位数相同
    for (int i = la - 1; i >= 0; i --){
        if (a[i] != b[i]) return a[i] > b[i];
    }
    // 大小相等，防止输出-0
    return true;
}

// 高精度减法：适用于两个正数相减，大数减去小数
void sub(int a[], int b[]){
    for (int i = 0; i < lc; i++){
        if (a[i] < b[i]){ // 借位
            a[i + 1]--;
            a[i] += 10;
        }
        c[i] = a[i] - b[i]; // 存差
    }
    // 处理前导0
    // 两数完全相等，最终lc = 0
    // 两数不相等，lc指向c[]中第一个不为0的数
    while (lc > 0 && c[lc] == 0) lc--;
}
int main(){
    cin >> sa >> sb;
    la = sa.size();
    lb = sb.size();
    lc = max(la, lb);
    for (int i = 0; i < la; i++) a[i] = sa[la - 1 - i] - '0';
    for (int i = 0; i < lb; i++) b[i] = sb[lb - 1 - i] - '0';
    if (!cmp(a, b)){
        cout << "-";
        swap(a, b);
    }
    sub(a, b);
    for (int i = lc; i >= 0; i--) cout << c[i];
    return 0;
}