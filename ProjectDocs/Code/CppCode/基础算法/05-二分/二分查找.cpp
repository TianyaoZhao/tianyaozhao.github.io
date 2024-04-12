// https://www.acwing.com/problem/content/791/
const int N = 1e5 + 10;
int a[N];
// 根据情况写check()函数

int find(int x){ // a[1~n]
    // 指向区间的两侧
    int l = 0, r = n + 1; 
    while(l + 1 < r){
        int mid = l + r >> 1;
        if(check(mid)) l = mid;
        else r = mid;
    }
    return l;
}

int find(int x){ // a[1~n]
    // 指向区间的两侧
    int l = 0, r = n + 1; 
    while(l + 1 < r){
        int mid = l + r >> 1;
        if(check(mid)) r = mid;
        else l = mid;
    }
    return r;
}