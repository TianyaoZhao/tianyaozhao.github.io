// https://www.acwing.com/problem/content/795/
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
const int N = 1e5 + 10;
int a[N], b[N], c[N];
string sa, sb;
int la, lb, lc;
// 高精度乘法:下标之和相等的处在数组c的同一个位置
void mul(int a[], int b[], int c[]){
  for(int i = 0; i < la; i ++){
    for(int j = 0; j < lb; j ++){
      c[i + j] += a[i] * b[j];
    }
  }
  // 进位
  for(int i = 0; i < lc; i ++){
    c[i + 1] += c[i] / 10;
    c[i] = c[i] % 10;
  }
  // 处理前导0
  while(lc > 0 && c[lc] == 0) lc --;
}
void solve(){
  cin >> sa >> sb;
  la = sa.size(), lb = sb.size();
  lc = la + lb;
  for(int i = 0; i < la; i ++) a[i] = sa[la - i - 1] - '0';
  for(int i = 0; i < lb; i ++) b[i] = sb[lb - i - 1] - '0';
  mul(a, b, c);
  for(int i = lc; i >= 0; i --) cout << c[i];
}
int main(){
  io_speed
  solve();
  return 0;
}