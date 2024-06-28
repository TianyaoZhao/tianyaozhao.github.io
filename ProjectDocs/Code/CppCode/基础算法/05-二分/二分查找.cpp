// https://www.acwing.com/problem/content/791/
const int N = 1e5 + 10;
int a[N], n;
// 根据情况写check()函数,哪个指针跳返回哪个
bool check(int mid){

}

// 寻找数组中值为x的项的下标
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

// 浮点数二分
const double M = 1e-4;
double find(int l, int r){
    while (r - l > M){ // 精度M比要求的精度高两个即可
        double mid = (l + r) / 2;
        if(check(mid)) l = mid;
        else r = mid;
    }
    return l;
}