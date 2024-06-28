// https://www.acwing.com/problem/content/3501/
// 计算两个日期相差多少天，相邻算2天 按天数枚举会TLE
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
int days[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}; 
bool check(int year, int month, int day){
    if(day == 0) return false;
    if(month > 12 || month < 1) return false;
    if(month != 2 ){
        if(day > days[month]) return false;
    }
    else{
        int leap = year % 4 == 0 && year % 100 || year % 400 == 0;
        if(day > days[2] + leap) return false;
    }
    return true;
}
void solve(){
    int st, ed;
    while(cin >> st >> ed){
        if(st > ed) swap(st, ed);
        int cnt = 0;
        for(int date = st; date <= ed; date ++){
            int year = date / 10000, month = date % 10000 / 100, day = date % 100;
            if(check(year, month, day)) cnt ++;
        }
        cout << cnt << endl;
    }

}
int main(){
    io_speed
    solve();
    return 0;
}