// https://www.acwing.com/problem/content/3721/
// 给定一个长度为 n 的数字串，向里面插入 k 个乘号，输出可以得到的最大结果
// 使用二进制枚举的方式，枚举含有k个1的状态，C_{n-1}^k
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
// const int N =
int n, k;
string s;
void solve(){
    cin >> n >> k >> s;
    // 二进制枚举2^(n-1)种状态
    LL res = 0;    
    int cnt = n - 1;
    for(int i = 0; i < (1 << cnt); i ++){
        // 判断位数
        int t = 0;
        for(int j = 0; j < cnt; j ++){
            t += i >> j & 1; //位为1的个数
        }
        // 1的个数为k, i是二进制状态
        if(t == k){
            // 
            LL p = 1;
            string str = s.substr(0, 1); // 取出第一个字符
            for(int j = 0; j < cnt; j ++){
                if(i >> j & 1){ // 当前位为1
                    // 取出当前存储的字符串，转为longlong
                    p *= stoll(str);
                    str = s.substr(j + 1, 1);
                }
                else{ // 当前位为0, 把下一个字符加上
                    str += s[j + 1];
                }
            }
            // 将结果乘上最后的字符
            p *= stoll(str);
            res = max(res, p);
        }
    }
    cout << res;
}
int main(){
    io_speed
    solve();
    return 0;
}