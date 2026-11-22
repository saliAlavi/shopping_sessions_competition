// fuzz: split_line<K> must agree with the exact scalar split<K> on arbitrary input
#define main pipeline_main
#include "../src/main.cpp"  // the pipeline, with its main() renamed
#undef main
#include <random>
int main() {
  std::mt19937_64 rng(12345);
  const char alpha[] = {'a', 'b', ',', '"', '\n', ',', '"', 'c', ',', '\r', ' ', ','};
  size_t total = 0, bad = 0, okc = 0;
  for (int iter = 0; iter < 400000; iter++) {
    int n = 1 + rng() % 300;
    std::string buf(1, '\n');
    int mode = rng() % 4;
    for (int i = 0; i < n; i++) {
      if (mode == 0) buf.push_back(alpha[rng() % sizeof alpha]);
      else {
        // mostly well-formed fields
        int len = (mode == 3 ? rng() % 60 : rng() % 12);
        bool q = rng() % 3 == 0;
        if (q) buf.push_back('"');
        for (int k = 0; k < len; k++) buf.push_back(rng() % 50 == 0 ? alpha[rng() % sizeof alpha] : 'x');
        if (q && rng() % 20) buf.push_back('"');
        buf.push_back(rng() % 7 == 0 ? '\n' : ',');
      }
    }
    buf.push_back('\n');
    size_t real = buf.size();
    buf.append(256, '\0');
    for (int K : {2, 4, 7}) {
      const char* p1 = buf.data() + 1;
      const char* p2 = buf.data() + 1;
      const char* end = buf.data() + real;
      while (p1 < end) {
        Fld f1[7], f2[7];
        bool ok1, ok2;
        if (K == 2) { p1 = split<2>(p1, f1, ok1); ok2 = split_line<2>(p2, f2); }
        else if (K == 4) { p1 = split<4>(p1, f1, ok1); ok2 = split_line<4>(p2, f2); }
        else { p1 = split<7>(p1, f1, ok1); ok2 = split_line<7>(p2, f2); }
        total++;
        okc += ok1;
        bool same = ok1 == ok2 && p1 == p2;
        if (same && ok1) for (int i = 0; i < K; i++) same = same && f1[i].p == f2[i].p && f1[i].n == f2[i].n;
        if (!same) {
          if (bad++ < 5) fprintf(stderr, "MISMATCH K=%d ok %d/%d next %ld/%ld\n", K, ok1, ok2, (long)(p1 - buf.data()), (long)(p2 - buf.data()));
          p2 = p1;
        }
      }
    }
  }
  printf("lines %zu ok %zu mismatches %zu\n", total, okc, bad);
  return bad != 0;
}
