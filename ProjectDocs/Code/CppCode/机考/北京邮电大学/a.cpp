#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
using namespace std;
const int N=15;
int n,k;
int w[N][N];
// 将矩阵a和b相乘，结果存放到c中。 
void mul(int c[][N],int a[][N],int b[][N]){
    static int tmp[N][N];
    memset(tmp,0,sizeof tmp);
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            for(int k=0;k<n;k++)
                tmp[i][j]+=a[i][k]*b[k][j];
    memcpy(c,tmp,sizeof tmp); 
}

int main(){
    cin>>n>>k;
    // 读入原始矩阵 
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            cin>> w[i][j];
    // 将res初始化为单位矩阵 
    int res[N][N]={0};
    for(int i=0;i<n;i++) res[i][i]=1;
    
    // 输出结果矩阵 
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++)
            cout<<res[i][j]<<" ";
        cout<<endl; 
    } 
    return 0;
}
