// https://www.luogu.com.cn/problem/P5587
// re 有个样例没过
# include <iostream>
# include <cstring>
# include <algorithm>
# include <string>
# include <stack>
# include <cmath>
using namespace std;
//  使用栈来存放文本串和输入串
stack <string> a, b;
double t;
int main(){
    string s;
    while(getline(cin, s)){
        if(s == "EOF") break;
        a.push(s);
    }
    while(getline(cin, s)){
        if(s == "EOF") break;
        b.push(s);
    }
    cin >> t;
    double cnt = 0;
    while(!a.empty()){
        string s1 = a.top(); a.pop();
        string s2 = b.top(); b.pop();
        // 预处理s1字符串, 将带有<的全部处理完，只保留正常的字符
        for(int i = 0; i < s1.size(); i ++){
            if(i == 0 && s1[0] == '<') s1.erase(0, 1), i -= 1;
            else if(s1[i] == '<') s1.erase(i - 1, 2), i -= 2;
        }
        // 预处理s2字符串, 将带有<的全部处理完，只保留正常的字符
        for(int i = 0; i < s2.size(); i ++){
            // 记住删完之后指针要回退
            if(i == 0 && s2[0] == '<') s2.erase(0, 1), i-= 1;
            else if(s2[i] == '<') s2.erase(i - 1, 2), i-= 2;
        }
        for(int i = 0; i < min(s1.size(), s2.size()); i ++){
            if(s1[i] == s2[i]) cnt ++;
        }
    }
    cout << round(cnt / (t / 60));
    return 0;
}