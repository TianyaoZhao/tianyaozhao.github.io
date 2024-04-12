// https://www.acwing.com/file_system/file/content/whole/index/content/4313/
# include <iostream>
# include <cstring>
# include <algorithm>
using namespace std;
const int N = 1e5 + 10;
int n, a[N];

void qsort(int l, int r){
    if(l == r) return;

    int i = l - 1, j = r + 1, x = a[l + r >> 1];
    while(i < j){
        do i ++; while(a[i] < x);   // 向右找>=x的数
        do j --; while(a[j] > x);   // 向左找<=x的数
        if(i < j) swap(a[i], a[j]);
    }

    qsort(l, j);
    qsort(j + 1, r);
}
int main(){
    cin >> n;
    for(int i= 0; i < n; i ++) cin >> a[i];
    qsort(0, n - 1);
    for(int i = 0; i < n; i ++) cout << a[i] << " ";
    return 0;
}