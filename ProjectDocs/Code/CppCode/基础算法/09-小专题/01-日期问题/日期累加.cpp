// https://www.acwing.com/problem/content/3576/
// 计算一个日期加上若干天后是什么日期。
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
int T;
int days[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}; 
inline bool check(int year, int month, int day){
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
    cin >> T;
    int y, m, d, a;
    while(T --){
        cin >> y >> m >> d >> a;
        int date = y * 10000 + m * 100 + d;
        int cnt = 0;
        while(true){
            int year = date / 10000, month = date % 10000 / 100, day = date % 100;
            if(check(year, month, day)){
                if(cnt == a){
                    printf("%04d-%02d-%02d\n", year, month, day);
                    break;
                }
                cnt ++;
                date ++;
            }
            else{
                date ++;
            }
        }
    }
}
int main(){
    io_speed
    solve();
    return 0;
}