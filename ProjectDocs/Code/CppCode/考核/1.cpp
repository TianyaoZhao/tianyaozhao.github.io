// https://leetcode.cn/problems/two-sum/description/
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
const int N = 1e4 + 10;
struct node{
    int val;
    int idx;
};
vector <node> nums;
int n, target;
bool cmp(node a, node b){
    return a.val < b.val;
}
void solve(){
    cin >> n >> target;
    for(int i = 0; i < n; i ++){
        int x; cin >> x;
        nums.push_back({x, i});
    }
    sort(nums.begin(), nums.end(), cmp);
    // for(int i = 0; i < n; i ++) cout << nums[i].val;

    for(int i = 0; i < nums.size(); i ++){
        int t = target - nums[i].val;

        int l = -1, r = nums.size();
        while(l + 1 < r){
            int mid = l + r >> 1;
            if(nums[mid].val >= t) r = mid;
            else l = mid;
        }
        if(nums[r].val == t){
            if(nums[i].idx == nums[r].idx){
                cout << nums[i].idx << " " << nums[r].idx + 1;
            }
            else{
                cout << nums[i].idx << " ";
                cout << nums[r].idx << endl;
            }
            break;
        }   
    }
}
int main(){
    io_speed
    solve();
    return 0;
}