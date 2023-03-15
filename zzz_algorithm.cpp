// いろいろ貼ってあるけど、ACL使った方がいいものもあります

#include <bits/stdc++.h>
using namespace std;

# define REP(i,n) for (int i=0;i<(n);++i)
# define rep(i,a,b) for(int i=a;i<(b);++i)
# define all(v) v.begin(),v.end()
# define showVector(v) REP(i,v.size()){cout << (v[i]) << " ";} cout << endl;

template<class T> inline bool chmin(T &a, T b){ if(a > b) { a = b; return true;} return false;}
template<class T> inline bool chmax(T &a, T b){ if(a < b) { a = b; return true;} return false;}
typedef long long ll;


// pq
template<class T>
using MaxHeap = std::priority_queue<T>;

template<class T>
using MinHeap = std::priority_queue<T, std::vector<T>, std::greater<T>>;

// 多次元 vector 生成
template<class T>
vector<T> make_vec(size_t a){
  return vector<T>(a);
}
template<class T, class... Ts>
auto make_vec(size_t a, Ts... ts){
  return vector<decltype(make_vec<T>(ts...))>(a, make_vec<T>(ts...));
}



template<typename T>
T gcd(T a, T b) {
  if(a < b) swap(a,b);
    
  if(b == 0) return a;
  return gcd(b, a % b);
}

ll lcm(ll a, ll b){
  ll g = gcd(a,b);
  return (a/g)*b;
}

// 素数判定 O(√n)
bool is_prime(ll n){
  if(n == 1) return false;
  for(ll i = 2; i * i <= n; i++){
    if(n % i == 0) return false;
  }
  return true;
}

// 約数列挙 O(√n)
vector<ll> divisor(ll n){
  vector<ll> res;
  for(ll i = 1; i * i <= n; i++){
    if(n % i == 0){
      res.push_back(i);
      if(i != n / i) res.push_back(n / i);
    }
  }
  return res;
}

template<typename T>

// エラトステネスの篩
// return lp[i] := iの最小素因数(lp[i] == iならば素数)
vector<int> SieveOfEratosthenes(int N = 10000000){
  vector<int> lp(N + 1, 0);
  vector<int> pr;
  for (int i=2; i<=N; ++i) {
    if (lp[i] == 0) {
      lp[i] = i;
      pr.push_back(i);
    }
    for (int j=0; j<(int)pr.size() && pr[j]<=lp[i] && i*pr[j]<=N; ++j)
      lp[i * pr[j]] = pr[j];
  }
  return lp;
}



ll mypow(ll x, ll n){
  if(n == 0)
    return 1;
 
  if(n % 2 == 0)
    return mypow(x * x, n / 2);
  else
    return x * mypow(x, n - 1);
}
 

// 素因数分解
long long MOD = 1000000000 + 7;

template<typename T>
map<T, ll> prime_factorize(T x){
  map<T, ll> res;

  while(x%2==0){
    x/=2;
    res[2]++;
  }

  for(ll i=3;i*i<=x;i+=2){
    while(x%i==0){
      x/=i;
      res[i]++;
    }
  }
  if(x!=1) res[x]++;
  return res;
}



// ランレングス圧縮
vector<pair<char,int>> run_comp(string S){
  vector<pair<char,int>> v;
  char now = S[0];
  int num = 1;
  char tmp = S[S.size()-1];
  for(int i = 1; i < S.size(); i++){
    tmp = S[i];
    if(now == tmp){
      num++;
    } else { 
      v.push_back(make_pair(now, num));
      num = 1;
      now = tmp;
    }
  }
  v.push_back(make_pair(tmp, num));
  return v;
}



/* 大文字を小文字に変換 */
char tolower(char c) {
  return (c + 0x20);
}

/* 小文字を大文字に変換 */
char toupper(char c) {
  return (c - 0x20);
}

# define REP(i,n) for (int i=0;i<(n);++i)
 
struct edge{ll to, cost;};
typedef pair<ll,ll> P;
struct graph{
  ll V;
  vector<vector<edge> > G;
  vector<ll> d;
 
  graph(ll n){
    init(n);
  }
 
  void init(ll n){
    V = n;
    G.resize(V);
    d.resize(V);
    REP(i,V){
      d[i] = INF;
    }
  }
 
  void add_edge(ll s, ll t, ll cost){
    edge e;
    e.to = t, e.cost = cost;
    G[s].push_back(e);
  }
 
  void dijkstra(ll s){
    REP(i,V){
      d[i] = INF;
    }
    d[s] = 0;
    priority_queue<P,vector<P>, greater<P> > que;
    que.push(P(0,s));
    while(!que.empty()){
      P p = que.top(); que.pop();
      ll v = p.second;
      if(d[v]<p.first) continue;
      for(auto e : G[v]){
        if(d[e.to]>d[v]+e.cost){
          d[e.to] = d[v]+e.cost;
          que.push(P(d[e.to],e.to));
        }
      }
    }
  }
};

// LCS
string LCS(string s, string t){
 	int n, m;
	n = s.length();
	m = t.length();

	// dp[i][j] : sのi文字目、tのj文字目のLCSの長さ
  vector<vector<int>> dp(n+1, vector<int>(m+1, 0));
 
	for(int i = 1; i <= n; i++){
		for(int j = 1; j <= m; j++){
			if(s[i-1] == t[j-1]){
				dp[i][j] = dp[i-1][j-1] + 1;
			}else{
				dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
			}
		}
	}
 
	// dp配列を逆からたどってLCSを作成する
	// LCSの長さが更新されたときの文字を追加していく
 
	string ans = "";
	int x = n;
	int y = m;
	while(x > 0 && y > 0){
		if(dp[x][y] == dp[x-1][y]){
			x--;
		}else if(dp[x][y] == dp[x][y-1]){
			y--;
		}else{
			x--;
			y--;
      ans += s[x];
		}
	}
 
	reverse(ans.begin(), ans.end());
  return ans;
}


// LCA
// ダブリングと根からの距離で頑張る

vector<vector<int>> parent;
map<int,int> dist;

int query(int x, int y){
  // xとyのLCAまでの距離を求めて、
  // x->LCA->y->xという閉路の長さを返す
  if(dist[x] < dist[y]){
    return query(y,x);
  }

  int in = x;
  int out = y;

  // cerr << "before: x: " << x << " y: " << y << endl;
  if(dist[x] != dist[y]){
    int up = dist[x] - dist[y];
    int log_up = log2(up);
    // xのup個上に
    for(ll k = log_up; k >= 0; k--){
      if((up>>k)&1){
        x = parent[k][x];
      }
    }
    // cerr << "after: x: " << x << " y: " << y << endl;
  }

  int lca;
  if(x == y){
    lca = x;
  } else {
    int height = parent.size();
    for(int i = height -1; i >= 0; i--){
      if(parent[i][x] != parent[i][y]){
        x = parent[i][x];
        y = parent[i][y];
      }
    }
    // cerr << "LCA: " << parent[0][x] << endl;
    lca = parent[0][x];
  }

  return lca;
}

int main(){
  int N;
  cin >> N;
  vector<vector<int>> v(N);
  int root;
  for(int i = 0; i < N; i++){
    int tmp;
    cin >> tmp;
    if(tmp == -1){
      root = i;
    } else { 
      tmp--;
      v.at(i).push_back(tmp);
      v.at(tmp).push_back(i);
    }
  }

  // 0からの距離
  dist[root] = 0;
  queue<int> q;
  q.push(root);

  int log_N = floor(log2(N));
  // cerr << N << " " << log_K << endl;
  parent.assign(log_N+1, vector<int>(N));

  parent[0][root] = -1;

  while(!q.empty()){
    int now = q.front();
    q.pop();

    for(int next: v.at(now)){
      if(dist.count(next) == 0){
        parent[0][next] = now;
        dist[next] = dist[now] + 1;
        q.push(next);
      }
    }
  }

  // ダブリング 
  for(int k = 0; k < log_N; k++){
    for(int i = 0; i < N; i++){
      if(parent[k][i] == -1){
        parent[k+1][i] = -1;
      } else { 
        parent[k+1][i] = parent[k][parent[k][i]];
      }
    }
  }


  int Q;
  cin >> Q;
  for(int i= 0; i < Q; i++){
    int tmp1,tmp2;
    cin >> tmp1 >> tmp2;
    tmp1--;
    tmp2--;
    int lca = query(tmp1,tmp2);
    string ans;
    if(tmp2 == lca){
      ans = "Yes";
    } else {
      ans = "No";
    }
    cout << ans << endl;

  }

}


// 2次元累積和
int main() {
  // 入力: H × W のグリッド
  int H, W; cin >> H >> W;
  vector<vector<int> > a(H, vector<int>(W));
  for (int i = 0; i < H; ++i) for (int j = 0; j < W; ++j) cin >> a[i][j];

  // 二次元累積和
  vector<vector<int> > s(H+1, vector<int>(W+1, 0));
  for (int i = 0; i < H; ++i)
    for (int j = 0; j < W; ++j)
      s[i+1][j+1] = s[i][j+1] + s[i+1][j] - s[i][j] + a[i][j];

  // クエリ [x1, x2) × [y1, y2) の長方形区域の和
  int Q; cin >> Q;
  for (int q = 0; q < Q; ++q) {
    int x1, x2, y1, y2;
    cin >> x1 >> x2 >> y1 >> y2;
    cout << s[x2][y2] - s[x1][y2] - s[x2][y1] + s[x1][y1] << endl;
  }
}  





// クラスカル法
typedef pair<int,int> pii;

long long MOD = 1000000000 + 7;

struct UnionFind {
  vector<int> data;
  UnionFind(int size) : data(size, -1) { }
  bool unionSet(int x, int y) {
    x = root(x); y = root(y);
    if (x != y) {
      if (data[y] < data[x]) swap(x, y);
      data[x] += data[y]; data[y] = x;
    }
    return x != y;
  }
  bool findSet(int x, int y) {
    return root(x) == root(y);
  }
  int root(int x) {
    return data[x] < 0 ? x : data[x] = root(data[x]);
  }
  int size(int x) {
    return -data[root(x)];
  }
};

int main(){
  cout << setprecision(10);
  int N,M; cin >> N >> M;
  vector<pair<int, pair<int,int>>> v(M);

  for(int i = 0; i < M; i++){
    int x,y,z; cin >> x >> y >> z;
    v[i] = make_pair(z, make_pair(x,y));
  }

  sort(v.begin(), v.end());


  UnionFind tree(N);
  vector<pair<int,int>> ans;
  ll min_val = 0;
  for(int i = 0; i < M; i++){
    ll weight = (ll)v[i].first;
    int s = v[i].second.first;
    int t = v[i].second.second;

    if(tree.root(s) == tree.root(t)){
      continue;
    } else { 
      min_val += weight;
      tree.unionSet(s,t);
    }
  }
  cout << min_val << endl;
}


//形式的冪級数

#define rep2(i, m, n) for (int i = (m); i < (n); ++i)
#define rep(i, n) rep2(i, 0, n)
#define drep2(i, m, n) for (int i = (m)-1; i >= (n); --i)
#define drep(i, n) drep2(i, n, 0)

template<class T>
struct FormalPowerSeries : vector<T> {
  using vector<T>::vector;
  using vector<T>::operator=;
  using F = FormalPowerSeries;

  F operator-() const {
    F res(*this);
    for (auto &e : res) e = -e;
    return res;
  }
  F &operator*=(const T &g) {
    for (auto &e : *this) e *= g;
    return *this;
  }
  F &operator/=(const T &g) {
    assert(g != T(0));
    *this *= g.inv();
    return *this;
  }
  F &operator+=(const F &g) {
    int n = (*this).size(), m = g.size();
    rep(i, min(n, m)) (*this)[i] += g[i];
    return *this;
  }
  F &operator-=(const F &g) {
    int n = (*this).size(), m = g.size();
    rep(i, min(n, m)) (*this)[i] -= g[i];
    return *this;
  }
  F &operator<<=(const int d) {
    int n = (*this).size();
    (*this).insert((*this).begin(), d, 0);
    (*this).resize(n);
    return *this;
  }
  F &operator>>=(const int d) {
    int n = (*this).size();
    (*this).erase((*this).begin(), (*this).begin() + min(n, d));
    (*this).resize(n);
    return *this;
  }
  F inv(int d = -1) const {
    int n = (*this).size();
    assert(n != 0 && (*this)[0] != 0);
    if (d == -1) d = n;
    assert(d > 0);
    F res{(*this)[0].inv()};
    while (res.size() < d) {
      int m = size(res);
      F f(begin(*this), begin(*this) + min(n, 2*m));
      F r(res);
      f.resize(2*m), internal::butterfly(f);
      r.resize(2*m), internal::butterfly(r);
      rep(i, 2*m) f[i] *= r[i];
      internal::butterfly_inv(f);
      f.erase(f.begin(), f.begin() + m);
      f.resize(2*m), internal::butterfly(f);
      rep(i, 2*m) f[i] *= r[i];
      internal::butterfly_inv(f);
      T iz = T(2*m).inv(); iz *= -iz;
      rep(i, m) f[i] *= iz;
      res.insert(res.end(), f.begin(), f.begin() + m);
    }
    return {res.begin(), res.begin() + d};
  }

  // // fast: FMT-friendly modulus only
  // F &operator*=(const F &g) {
  //   int n = (*this).size();
  //   *this = convolution(*this, g);
  //   (*this).resize(n);
  //   return *this;
  // }
  // F &operator/=(const F &g) {
  //   int n = (*this).size();
  //   *this = convolution(*this, g.inv(n));
  //   (*this).resize(n);
  //   return *this;
  // }

  // // naive
  // F &operator*=(const F &g) {
  //   int n = (*this).size(), m = g.size();
  //   drep(i, n) {
  //     (*this)[i] *= g[0];
  //     rep2(j, 1, min(i+1, m)) (*this)[i] += (*this)[i-j] * g[j];
  //   }
  //   return *this;
  // }
  // F &operator/=(const F &g) {
  //   assert(g[0] != T(0));
  //   T ig0 = g[0].inv();
  //   int n = (*this).size(), m = g.size();
  //   rep(i, n) {
  //     rep2(j, 1, min(i+1, m)) (*this)[i] -= (*this)[i-j] * g[j];
  //     (*this)[i] *= ig0;
  //   }
  //   return *this;
  // }

  // sparse
  F &operator*=(vector<pair<int, T>> g) {
    int n = (*this).size();
    auto [d, c] = g.front();
    if (d == 0) g.erase(g.begin());
    else c = 0;
    drep(i, n) {
      (*this)[i] *= c;
      for (auto &[j, b] : g) {
        if (j > i) break;
        (*this)[i] += (*this)[i-j] * b;
      }
    }
    return *this;
  }
  F &operator/=(vector<pair<int, T>> g) {
    int n = (*this).size();
    auto [d, c] = g.front();
    assert(d == 0 && c != T(0));
    T ic = c.inv();
    g.erase(g.begin());
    rep(i, n) {
      for (auto &[j, b] : g) {
        if (j > i) break;
        (*this)[i] -= (*this)[i-j] * b;
      }
      (*this)[i] *= ic;
    }
    return *this;
  }

  // multiply and divide (1 + cz^d)
  void multiply(const int d, const T c) { 
    int n = (*this).size();
    if (c == T(1)) drep(i, n-d) (*this)[i+d] += (*this)[i];
    else if (c == T(-1)) drep(i, n-d) (*this)[i+d] -= (*this)[i];
    else drep(i, n-d) (*this)[i+d] += (*this)[i] * c;
  }
  void divide(const int d, const T c) {
    int n = (*this).size();
    if (c == T(1)) rep(i, n-d) (*this)[i+d] -= (*this)[i];
    else if (c == T(-1)) rep(i, n-d) (*this)[i+d] += (*this)[i];
    else rep(i, n-d) (*this)[i+d] -= (*this)[i] * c;
  }

  T eval(const T &a) const {
    T x(1), res(0);
    for (auto e : *this) res += e * x, x *= a;
    return res;
  }

  F operator*(const T &g) const { return F(*this) *= g; }
  F operator/(const T &g) const { return F(*this) /= g; }
  F operator+(const F &g) const { return F(*this) += g; }
  F operator-(const F &g) const { return F(*this) -= g; }
  F operator<<(const int d) const { return F(*this) <<= d; }
  F operator>>(const int d) const { return F(*this) >>= d; }
  F operator*(const F &g) const { return F(*this) *= g; }
  F operator/(const F &g) const { return F(*this) /= g; }
  F operator*(vector<pair<int, T>> g) const { return F(*this) *= g; }
  F operator/(vector<pair<int, T>> g) const { return F(*this) /= g; }
};

using mint = modint1000000007;
using fps = FormalPowerSeries<mint>;
using sfps = vector<pair<int, mint>>;

int main() {
  int n, k; cin >> n >> k;

  fps f = {1, -1};
  f.resize(n+1);

  f *= sfps{{1, 1}, {k, -1}};
  f /= sfps{{0, 1}, {1, -2}, {k+1, 1}};

  cout << f[n].val() << '\n';
}


// segtree 
// range min query
ll op(ll a, ll b){
  return min(a,b);
}
ll e(){
  return INF;
}

int main(){
  int N = 10;
  segtree<ll,op,e> tree(N);
}