// https://www.acwing.com/problem/content/805/
/* 给定n个区间，合并有交集的区间，输出合并后的区间个数 */
# include <iostream>
# include <cstring>
# include <algorithm>
# include <vector>
using namespace std;
typedef pair<int, int> PII;
vector<PII> segs;
int n;

void merge(){
    vector<PII> res;
    // 对所有区间按左端点从小到大排序
    sort(segs.begin(), segs.end());
    // 定义维护的区间起点和终点
    int st = -2e9, ed = -2e9;
    // 取区间片段
    for(auto seg:segs){
        if(ed < seg.first){ // 如果维护的区间和当前区间是分离的
            if(st != -2e9) res.push_back({st, ed}); // 保存结果
            st = seg.first;
            ed = seg.second;
        }
        else{ // 如果维护区间和当前区间有交集
            ed = max(ed, seg.second);
        }
    }
    // 最后把最后一段维护的区间加到res中
    res.push_back({st, ed});
    segs = res;
}


int main(){
    cin >> n;
    while(n --){
        int l, r;
        cin >> l >> r;
        segs.push_back({l, r});
    }
    merge();
    cout << segs.size();
    return 0;
}