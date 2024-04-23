// https://www.luogu.com.cn/problem/P1028
# include <iostream>
# include <cstring>
# include <algorithm>
using namespace std;
// 暴力递归
// const int N;
// int n, cnt;
// void dfs(int x){
//     for(int i = 1; i <= x / 2; i ++){
//         cnt ++;
//         dfs(i);
//     }
// }
// int main(){
//     cin >> n;
//     dfs(n);
//     cout << cnt + 1;
//     return 0;
// }

// 递推公式
// f[i]:表示答案的个数，当序列的最大值为i时
// 思考转移：
// 若 x = 2n + 1, 则f[2n + 1]的答案，绝大多数可以由f[2n]得到， f[2n]是从1~n枚举，f[2n + 1]是从1~n枚举
// 若 x = 2n, 则f[2n]的答案，绝大多数可以由f[2n - 1]得到, f[2n - 1]是从 1 ~ n - 1枚举，f[2n]是从1 ~ n 枚举少了 f[n]的部分
// 递推公式
// f[2n+1] = f[2n], f[2n] = f[2n-1]+f[n];
const int N = 1e3 + 10;
int f[N];
int n;
int main(){
    cin >> n;
    f[0] = 1;
    f[1] = 1;
    for(int i = 1; i <= n; i ++){
        if(i % 2 == 0) f[i] = f[i - 1] + f[i / 2];
        else f[i] = f[i - 1];
    }
    cout << f[n];
    return 0;
}