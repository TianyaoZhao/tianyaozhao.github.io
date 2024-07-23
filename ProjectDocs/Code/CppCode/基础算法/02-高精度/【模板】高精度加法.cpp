// https://www.acwing.com/problem/content/description/793/
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
const int N =  1e5 + 10;
int a[N], b[N], c[N];
string sa, sb;
int la, lb, lc;
void add(int a[], int b[], int c[]){
  for(int i = 0; i < lc; i ++){
    c[i] += a[i] + b[i];     // 累加
    c[i + 1] += c[i] / 10;   // 进位
    c[i] = c[i] % 10;        // 存余
  }
  if(c[lc]) lc ++;           // 最高位有进位
}
void solve(){
  cin >> sa >> sb;
  la = sa.size(), lb = sb.size();
  lc = max(la, lb);

  // 反向存储，确保低位下标从0开始
  for(int i = 0; i < la; i ++) a[i] = sa[la - i - 1] - '0';  //减0
  for(int i = 0; i < lb; i ++) b[i] = sb[lb - i - 1] - '0';

  add(a, b, c);

  for(int i = lc - 1; i >= 0; i --) cout << c[i];
}
int main(){
  io_speed
  solve();
  return 0;
}