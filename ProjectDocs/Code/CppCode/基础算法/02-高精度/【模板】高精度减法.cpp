// https://www.acwing.com/problem/content/794/
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
// 比较 a > b
bool cmp(int a[], int b[]){
  // 位数不相同
  if(la != lb) return la > lb;
  // 位数相同
  for(int i = lc - 1; i >= 0; i --){
    if(a[i] != b[i]) return a[i] > b[i];
  }
  // 完全相等，防止输出-0
  return true;  
}

void sub(int a[], int b[], int c[]){
  for(int i = 0; i < lc; i ++){
    if(a[i] < b[i]){
      a[i + 1] --;      // 借位
      a[i] += 10;    
    }
    c[i] = a[i] - b[i]; // 存差
  }
  // 去除前导0, 此时lc指向第一个不为0的数，如果答案是0,c[0] = 0, lc = 0;
  while(lc > 0 && c[lc] == 0) lc --;
}
void solve(){
  cin >> sa >> sb;
  la = sa.size();
  lb = sb.size();
  lc = max(la, lb);
  
  for(int i = 0; i < la; i ++) a[i] = sa[la - i - 1] - '0';
  for(int i = 0; i < lb; i ++) b[i] = sb[lb - i - 1] - '0';
  
  // 先判断大小关系
  if(!cmp(a, b)){
    swap(a, b);
    cout << '-';
  }
  sub(a, b, c);

  for(int i = lc; i >= 0; i --) cout << c[i];

}
int main(){
  io_speed
  solve();
  return 0;
}