// (a, b)和(b, a%b)的最大公约数相同
// d是(a, b)  的公约数, d|a, d|b,  => d|a-kb
// d是(b, a%b)的公约数, d|b, d|a%b => d|a-kb

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
LL gcd(int a, int b){
    if(b == 0) return a;
    else return gcd(b, a % b);
}

LL lcm(int a, int b){
    return a * b / gcd(a, b);
}
void solve(){

}
int main(){
    io_speed
    solve();
    return 0;
}