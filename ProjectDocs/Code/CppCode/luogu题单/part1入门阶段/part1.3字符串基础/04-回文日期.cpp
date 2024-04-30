// https://www.luogu.com.cn/problem/P2010
// 判断两个日期之间有多少个回文数
# include <iostream>
# include <cstring>
# include <algorithm>
# include <string>
using namespace std;
const int N = 10;
int days[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
int date1, date2;
bool check(int year, int month, int day){
    if(month < 1 || month > 12) return false;

    if(day == 0) return false;

    // 特判二月闰年的情况
    if(month != 2){
        if(day > days[month]) return false;
    }
    else{
        // 四年一闰百年不闰or四百年一闰
        int leap = year % 4 == 0 && year % 100 || year % 400 == 0;
        if(day > days[2] + leap) return false;
    }
    return true;
}

bool ishw(int date){
    int a[N] = {};
    for(int i = 1; i <= 8; i ++){
        a[i] = date % 10;
        date = date / 10;
    } 
    for(int i = 1; i <= 8; i ++){
        if(a[i] != a[8 - i + 1]) return false;
    }
    return true;
}
int main(){
    cin >> date1 >> date2;
    int cnt = 0;
    for(int i = date1; i <= date2; i ++){
        int year = i / 10000, month = i % 10000 / 100, day = i % 100;
        if(check(year, month, day)){
            if(ishw(i)) cnt ++;
        }
    }
    cout << cnt;
    return 0;
}