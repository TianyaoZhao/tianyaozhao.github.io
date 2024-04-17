// https://www.luogu.com.cn/problem/P1308
// 给定一个单词，请你输出它在给定的文章中出现的次数和第一次出现的位置。注意：匹配单词时，不区分大小写
# include <iostream>
# include <cstring>
# include <algorithm>
# include <string>
using namespace std;
const int N = 1e6 + 10;
int k;
string s, so, t;
string a[N];
// 检查两串是否是相等的,除第一个字符大小写不敏感
bool check(string so, string t){
    if(so.size() == t.size()){   // 长度相同
        if(so[0] - t[0] == 0 || so[0] - t[0] == 32 || so[0] - t[0] == -32){
            for(int i = 1; i < t.size(); i ++){
                if(so[i] != t[i]) return false; // 字母不相同
            }
            return true; // 都相同
        }
        else return false; // 第一位不相同
    }
    else return false; // 长度不同
}
// 检查两串是否是相等的,大小写不敏感
bool check2(string so, string t){
    if(so.size() == t.size()){   // 长度相同
            for(int i = 0; i < t.size(); i ++){
                if(so[i] - t[i] == 0 || so[i] - t[i] == 32 || so[i] - t[i] == -32) continue;
                else return false; // 字母不相同
            }
            return true; // 都相同
    }
    else return false; // 长度不同
}
int main(){
    cin >> so;
    getchar();
    getline(cin, s);
    // 存储单词
    int cnt = 0, idx = 0x3f3f3f3f;
    for(int i = 0; i < s.size(); i++){
        // 等于空格就看看是否是空串，不是空串就输出出来
        if(s[i] == ' '){
            if(t != ""){
                 a[k ++] = t;
                 if(check2(so, t)){ // 匹配
                    cnt ++;
                    idx = min(idx, int(i - t.size()));
                 }
            }
            t = "";
        }
        // 等于其他字符就把t放入临时字符串中
        else t.push_back(s[i]);
    }
    // 打印最后一个单词
    if(t != ""){
        a[k ++] = t;
        if(check2(so, t)){
            cnt ++;
            idx = min(idx, int(s.size() - t.size()));
        }
    }
    // for(int i = 0; i < k; i ++) cout << a[i] << endl;
    if(cnt == 0) cout << -1;
    else cout << cnt << " " << idx;
    return 0;
}