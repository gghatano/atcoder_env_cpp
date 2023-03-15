#include<bits/stdc++.h>
#include<atcoder/all>
using namespace atcoder;
using namespace std;


template<class T> inline bool chmin(T &a, T b){ if(a > b) { a = b; return true;} return false;}
template<class T> inline bool chmax(T &a, T b){ if(a < b) { a = b; return true;} return false;}

typedef long long ll;
typedef pair<int,int> pii;

// mint
using mint = static_modint<1000000007>;
using mint = static_modint<998244353>;
// ll int
ll INF = numeric_limits<ll>::max() / 2;

int main(){
  // set precision (10 digit)
  cout << setprecision(10);
}
