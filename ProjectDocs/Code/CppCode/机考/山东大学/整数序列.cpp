// https://www.acwing.com/problem/content/3720/
// 很多整数可以由一段连续的正整数序列（至少两个数）相加而成 25 = 3 + 4 + 5 + 6 + 7 = 12 + 13  
// 输入一个整数 N，输出 N 的全部正整数序列，如果没有则输出 NONE
// 双指针查满足条件的区间,等差数列求和
// 也可以枚举起点+二分查找终点
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
int n;
void solve(){
    cin >> n;
    int i = 1, j = 1, len = 1;
    bool flag = false;
    while(i < n && j < n){
        // 计算区间和
        int num = (i + j) * len / 2;
        if(num < n){
            j ++;
            len ++;
        }
        else if(num > n){
            i ++;
            // j不动
            len --;
        }
        else{ // 相等
            for(int l = i; l <= j; l ++) cout << l <<" ";
            cout << endl;
            flag = true;
            i = i + 1;
            j = i;
            len = 1;
        }
    }
    if(!flag) cout << "NONE";
    
}
int main(){
    io_speed
    solve();
    return 0;
}