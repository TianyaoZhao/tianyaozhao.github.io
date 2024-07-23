// https://www.acwing.com/problem/content/804/
/* 给定一个无限长的数轴，求某一个区间和是多少 */
# include <iostream>
# include <cstring>
# include <algorithm>
# include <vector>
using namespace std;
const int N = 3e5 + 10;      // n次操作，m次查询，需要的下标个数为n+2m
int a[N], s[N];
typedef pair<int, int> PII;
vector <PII> add, query;     // 存放两种操作：相加x c 查询 l r 参数
vector <int> alls;           // 存放待离散化的数值, 将原本数轴上的坐标存入vector，其值映射为vector数组的下标值 
int n, m;

//二分查找 返回离散化后的数组下标 个二分的板子不常用
int find(int x){
    int l = 0, r=alls.size()-1;
    while(l < r)
    {
        int mid = l + r >> 1;
        if(alls[mid] >= x) r = mid;
        else l = mid + 1;
    }
    //因为此题还要求前缀和，所以把映射下标整体右移一位
    return r + 1;
}
int main(){
    cin >> n >> m;
    for(int i = 0; i < n; i ++){
        int x, c;
        cin >> x >> c;
        alls.push_back(x);         // 待离散化的下标
        add.push_back({x, c});     // 相加的操作数
    }
    for(int i = 0; i < m; i ++){
        int l, r;
        cin >> l >> r;
        alls.push_back(l);          // 待离散化的下标
        alls.push_back(r);
        query.push_back({l, r});    // 查询的操作数
    }

    // 对alls 数组去重，因为可能有多个重复的下标
    sort(alls.begin(), alls.end());
    // unique()返回alls数组去重后的最后一个元素的下标
    // alls.erase() 删除元素
    alls.erase(unique(alls.begin(), alls.end()), alls.end());

    // 处理相加
    for(auto item:add){
        int x = find(item.first);  // 寻找离散化后的数组下标
        int c = item.second;
        a[x] += c;
    }

    // 构造前缀和(注意是alls.size(),这个范围要比a[x].size大)
    for(int i = 1; i <= alls.size(); i ++) s[i] = s[i - 1] + a[i];

    // 处理查询
    for(auto item:query){
        int l = find(item.first); // 寻找离散化后的数组下标
        int r = find(item.second); 
        cout << s[r] - s[l - 1] << endl;
    }
    return 0;
}