// https://www.acwing.com/problem/content/803/
// 求二进制数第k位数字：n >> k & 1
// 返回n的最后一位1：lowbit(n) = n & -n  lowbit(101000)=1000
# include <iostream>
# include <cstring>
# include <algorithm>
using namespace std;
int n;
int lowbit(int x){
    return x & -x;
}
int main(){
    cin >> n;
    while(n --){
        int x; cin >> x;
        int cnt = 0;
        while(x){
            x = x - lowbit(x);
            cnt ++;
        }
        cout << cnt << " ";
    }
    return 0;
}