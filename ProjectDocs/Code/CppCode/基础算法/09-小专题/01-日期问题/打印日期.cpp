// https://www.acwing.com/problem/content/3610/
// 给出年份和一年中的第d天，计算d天是几月几号
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
    int y, d;
    while(cin >> y >> d){
        int cnt  = 1, date = y * 10000 + 0101;
        while(true){
            int year = date / 10000, month = date % 10000 / 100, day = date % 100;
            if(check(year, month, day)){
                if(cnt == d){
                    // if(year < 1000) cout << "0";
                    // if(year < 100) cout << "0";
                    // if(year < 10) cout << "0";

                    // cout << year << "-" ;
                    // if(month < 10) cout << "0";
                    // cout << month << "-";
                    // if(day < 10) cout << "0";
                    // cout << day << endl; 
                    // 格式输出
                    printf("%04d-%02d-%02d\n",year ,month, day);
                    break;
                }
                cnt ++;
                date ++;
            }
            else date ++;
        }
    }
    
}
int main(){
    io_speed
    solve();
    return 0;
}