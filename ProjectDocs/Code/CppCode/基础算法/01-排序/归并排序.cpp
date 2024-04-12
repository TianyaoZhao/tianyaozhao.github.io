// https://www.acwing.com/problem/content/789/
# include <iostream>
# include <cstring>
# include <algorithm>
using namespace std;
const int N = 1e5 + 10;
int n, a[N], b[N];

void msort(int l, int r){
    if(l == r) return;
   
    // 向下拆分
    int mid = l + r >> 1;
    msort(l, mid);
    msort(mid + 1, r);

    // 回溯归并
    int i = l, j = mid + 1, k = l;
    while(i <= mid && j <= r){
        if(a[i] <= a[j]) b[k ++] = a[i ++];
        else b[k ++] = a[j ++];
    }
    while(i <= mid) b[k ++] = a[i ++];
    while(j <= r)   b[k ++] = a[j ++];

    for(int i = l; i <= r; i ++) a[i] = b[i];
}
int main(){
    cin >> n;
    for(int i= 0; i < n; i ++) cin >> a[i];
    msort(0, n - 1);
    for(int i = 0; i < n; i ++) cout << a[i] << " ";
    return 0;
}