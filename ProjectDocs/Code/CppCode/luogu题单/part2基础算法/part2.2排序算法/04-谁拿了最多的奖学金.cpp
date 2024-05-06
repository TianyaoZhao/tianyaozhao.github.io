// https://www.luogu.com.cn/problem/P1051
# include <iostream>
# include <cstring>
# include <algorithm>
# include <string>
# include <cmath>
# include <stack>
# include <queue>
# include <vector>
# define io_speed ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
# define endl '\n'
using namespace std;
typedef pair <int, int> PII;
typedef long long LL;
// const int N = 
int n, sum;
int check(int avg, int py, char isg, char isw, int num){
    int val = 0;
    if(avg > 80 && num >= 1) val += 8000;
    if(avg > 85 && py > 80) val += 4000;
    if(avg > 90) val += 2000;
    if(avg > 85 && isw == 'Y') val += 1000;
    if(py > 80 && isg =='Y') val += 850;
    return val;
}
void solve(){
    cin >> n;
    string name;
    int avg, py;
    char isg, isw;
    int num;
    int res = 0;
    string tname = "";
    while(n --){
        cin >> name >> avg >> py >> isg >> isw >> num;
        int val = check(avg, py, isg, isw, num);
        sum += val;
        if(val > res){
            tname = name;
            res = val;
        }
    }
    cout << tname << endl;
    cout << res << endl;
    cout << sum;
}
int main(){
    io_speed
    solve();
    return 0;
}