// https://www.acwing.com/problem/content/3394/
// 输入年、月、日，计算该天是本年的第几天
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
int year, month, day;
// 判断日期合法的函数
int days[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
bool check(int year, int month, int day){
    // 日期不为0
    if(day == 0) return false;
    // 12个月
    if(month < 1 || month > 12) return false;
    // 不能超每月天数
    if(month != 2){
        if(day > days[month]) return false;
    }
    // 特判2月
    else{
        int leap = year % 4 == 0 && year % 100 || year % 400 == 0;
        if(day > days[2] + leap) return false;
    }
    return true;
}
void solve(){
    while(cin >> year >> month >> day){
        int cnt = 0;
        int st = year * 10000 + 0101;
        int ed = year * 10000 + month * 100 + day;
        for(int date = st; date <= ed; date ++){
            int a = date / 10000, b = date % 10000 / 100, c = date % 100;
            if(check(a, b, c)) cnt ++;
        }
        cout << cnt << endl;
    }
}
int main(){
    io_speed
    solve();
    return 0;
}