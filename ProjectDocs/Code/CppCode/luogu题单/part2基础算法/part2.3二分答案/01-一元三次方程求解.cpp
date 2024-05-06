// https://www.luogu.com.cn/problem/P1024
// ax^3 + bx^2 + cx + d = 0 求实根
// 告诉根与根的绝对值之差 <=1 且在-100~100之间
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
double a, b, c, d;
double f(double x){
    return a * x * x * x + b * x * x + c * x + d; 
}
double find(double l, double r){
    while(r - l > 1e-4){
        double mid  = (l + r) / 2;
        if(f(l) * f(mid) < 0) r = mid;
        else l = mid;
    }
    return l;
}
void solve(){
    cin >> a >> b >> c >> d;
    for(double i = -100; i <= 100; i ++){
        if(f(i) == 0) cout << fixed << setprecision(2) << i << " ";
        // 说明在这个范围内有根, 用浮点数二分找根
        if(f(i) * f(i + 1) < 0) cout << fixed << setprecision(2) <<  find(i, i + 1) << " ";
    }
}
int main(){
    io_speed
    solve();
    return 0;
}
