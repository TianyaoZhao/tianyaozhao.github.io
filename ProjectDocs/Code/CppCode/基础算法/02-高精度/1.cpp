# include <iostream>
# include <cstring>
# include <algorithm>
# include <string>
using namespace std;
const int N = 100 + 10;
int a[N], b[N], c[N], d[N], e[N];
int la, lb, lc, ld, le;
string sa, sb;
void mul_1(int a[]){
  for(int i = 0; i < la; i ++){
    for(int j = 0; j < la; j ++){
      c[i + j] += a[i] * a[j];
    }
  }
  for(int i = 0; i < lc; i ++){
    c[i + 1] += c[i] / 10;
    c[i] %= 10;
  }
  while(lc > 0 && c[lc] == 0) lc --;
}
void mul_2(int b[]){
  for(int i = 0; i < lb; i ++){
    for(int j = 0; j < lb; j ++){
      d[i + j] += b[i] * b[j];
    }
  }
  for(int i = 0; i < ld; i ++){
    d[i + 1] += d[i] / 10;
    d[i] %= 10;
  }
  while(ld > 0 && d[ld] == 0) ld --;
}
bool cmp(int c[], int d[]){
  if(lc != ld) return lc > ld;
  for(int i = lc; i >= 0; i --){
    if(c[i] != d[i]) return c[i] > d[i];
  }
  return true;
}

void sub(int c[], int d[]){
  for(int i = 0; i < le; i ++){
    if(c[i] < d[i]){
      c[i + 1] --;
      c[i] += 10;
    }
    e[i] = c[i] - d[i];
  }
  while(le > 0 && e[le] == 0) le --;
}


int main(){
  cin >> sa >> sb;
  la = sa.size();
  lb = sb.size();
  if(sa[0] == '-'){
    for(int i = 0; i < la - 1; i ++) a[i] = sa[la - 1 -i] - '0';
    la --;
  }
  else{
    for(int i = 0; i < la; i ++) a[i] = sa[la - 1 -i] - '0';
  }

  if(sb[0] == '-'){
    for(int i = 0; i < lb - 1; i ++) b[i] = sb[lb - 1 - i] - '0';
    lb --;
  }
  else{
    for(int i = 0; i < lb; i ++) b[i] = sb[lb - 1 -i] - '0';
  }
  lc = la + la;
  ld = lb + lb;
  mul_1(a);
  mul_2(b);
  // for(int i = lc; i >= 0; i --) cout << c[i];
  // cout << endl;
  // for(int i = ld; i >= 0; i --) cout << d[i];

  if(!cmp(c, d)){
    cout << "-";
    swap(c, d);
  }
  le = max(lc, ld);
  sub(c, d);
  for(int i = le; i >= 0; i --) cout << e[i];
  return 0;
}