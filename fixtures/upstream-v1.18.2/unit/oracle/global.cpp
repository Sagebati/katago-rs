// Oracle for the rest of core/global (the functions the pilot left out). Links against the
// unmodified upstream sources of KataGo v1.16.5.
// The Rust mirror is crates/katago-core/tests/global.rs: keep the two in the same order.
// Build (from the repo root, output outside the repo; upstream untouched; asserts on):
//   U=upstream/KataGo/cpp; g++ -std=c++17 -O1 -I$U -o /tmp/global_oracle \
//     fixtures/upstream-v1.18.2/unit/oracle/global.cpp $U/core/global.cpp
//   /tmp/global_oracle > fixtures/upstream-v1.18.2/unit/oracle/global.txt
// Generated with g++ 16.2.1 / glibc on x86_64 Linux.
#include <cstdio>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "core/global.h"
using namespace std;

static string q(const string& s) { return "[" + s + "]"; }
static string hx(const string& s) {
  string r;
  char b[4];
  for(unsigned char c : s) { snprintf(b, sizeof(b), "%02X", c); r += b; }
  return r;
}
// Well-formed UTF-8 (no overlongs, surrogates or code points above U+10FFFF), as Rust's from_utf8.
static bool validUtf8(const string& s) {
  size_t i = 0, n = s.size();
  while(i < n) {
    unsigned char c = s[i];
    size_t len; unsigned int cp;
    if(c < 0x80) { i++; continue; }
    else if(c >= 0xC2 && c <= 0xDF) { len = 2; cp = c & 0x1F; }
    else if(c >= 0xE0 && c <= 0xEF) { len = 3; cp = c & 0x0F; }
    else if(c >= 0xF0 && c <= 0xF4) { len = 4; cp = c & 0x07; }
    else return false;
    if(i + len > n) return false;
    for(size_t k = 1; k < len; k++) {
      unsigned char d = s[i+k];
      if((d & 0xC0) != 0x80) return false;
      cp = (cp << 6) | (d & 0x3F);
    }
    if((len == 3 && cp < 0x800) || (len == 4 && (cp < 0x10000 || cp > 0x10FFFF)) || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
    i += len;
  }
  return true;
}

int main() {
  // charToString
  for(int c : {0, (int)'a', (int)' ', (int)'~', (int)'0'}) {
    string s = Global::charToString((char)c);
    cout << "charToString " << c << " " << s.size() << " " << q(s) << "\n";
  }
  // chopPrefix / chopSuffix
  vector<pair<string,string>> chops = {{"abcdef","abc"},{"abcdef",""},{"abc","abcdef"},{"abcdef","bcd"},{"",""},{"abcdef","abcdef"},{"abcdef","def"}};
  for(auto& p : chops) {
    try { cout << "chopPrefix " << q(p.first) << q(p.second) << " -> " << q(Global::chopPrefix(p.first,p.second)) << "\n"; }
    catch(const StringError& e) { cout << "chopPrefix " << q(p.first) << q(p.second) << " throws " << q(e.what()) << "\n"; }
    try { cout << "chopSuffix " << q(p.first) << q(p.second) << " -> " << q(Global::chopSuffix(p.first,p.second)) << "\n"; }
    catch(const StringError& e) { cout << "chopSuffix " << q(p.first) << q(p.second) << " throws " << q(e.what()) << "\n"; }
  }
  // concat
  {
    const char* arr[] = {"a", "", "bc", "def"};
    for(size_t len = 0; len <= 4; len++) {
      cout << "concat_len " << len << " " << q(Global::concat(arr, len, ", ")) << " " << q(Global::concat(arr, len, "")) << "\n";
    }
    vector<string> v = {"x", "yy", "", "zzz"};
    cout << "concat " << q(Global::concat(v, "|")) << " " << q(Global::concat(vector<string>(), "|")) << "\n";
    for(size_t s = 0; s <= 4; s++)
      for(size_t e = s; e <= 4; e++)
        cout << "concat_start_end " << s << " " << e << " " << q(Global::concat(v, "-", s, e)) << "\n";
    set<string> st = {"b", "a", "c", "aa", "B"};
    cout << "concat_set " << q(Global::concat(st, ",")) << " " << q(Global::concat(set<string>(), ",")) << "\n";
  }
  // isDigits / parseDigits
  vector<string> digs = {"", "0", "7", "123", "000000000", "999999999", "1234567890", "2147483647", "12a", "a12", "-1", " 1", "1 ", "00000000000", "0x1", "١"};
  for(const string& s : digs) {
    cout << "isDigits " << q(s) << " " << Global::isDigits(s);
    try { cout << " parseDigits " << Global::parseDigits(s); }
    catch(const StringError& e) { cout << " parseDigits throws " << q(e.what()); }
    cout << "\n";
    for(size_t a = 0; a <= s.size() + 1; a++) {
      for(size_t b = 0; b <= s.size() + 1; b++) {
        if(a > s.size()) continue; // substr(a,...) with a > size throws std::out_of_range (not mirrored)
        cout << "isDigits_se " << q(s) << " " << a << " " << b << " " << Global::isDigits(s, a, b);
        try { cout << " parseDigits_se " << Global::parseDigits(s, a, b); }
        catch(const StringError& e) { cout << " parseDigits_se throws " << q(e.what()); }
        cout << "\n";
      }
    }
  }
  // stringCharsAllAllowed
  vector<pair<string,string>> allowed = {{"abc","abc"},{"abc","ab"},{"","x"},{"aaa","a"},{"a b","ab "},{"xyz",""}};
  for(auto& p : allowed)
    cout << "stringCharsAllAllowed " << q(p.first) << q(p.second) << " " << Global::stringCharsAllAllowed(p.first, p.second.c_str()) << "\n";
  // stripComments
  vector<string> comments = {"", "abc", "a#b", "#", "x = 1 # c\ny = 2\n#full\n\nz=3", "no newline # end", "a\nb#\n", "\n\n#\n"};
  for(const string& s : comments)
    cout << "stripComments " << q(s) << " -> " << q(Global::stripComments(s)) << "\n";
  // readKeyValues
  vector<string> kvs = {"", "a=1", "a = 1, b= 2\nc=3", "a=1\na=2", "=1", "a=", "a", "a=1,,b=2", "x=y=z", " k = v w , m=n\n\n,\np=q", "a=1\r\nb=2", "a=1,b"};
  for(const string& s : kvs) {
    cout << "readKeyValues " << q(s) << " ->";
    try {
      map<string,string> m = Global::readKeyValues(s);
      for(auto& kv : m) cout << " " << q(kv.first) << "=" << q(kv.second);
    }
    catch(const StringError& e) { cout << " throws " << q(e.what()); }
    cout << "\n";
  }
  // readMem
  vector<string> mems = {"", "1", "12", "16G", "16GB", "256K", "256KB", "3M", "3MB", "1T", "1TB", "2P", "2PB", "100B", "100", "K", "KB", "1.5G", "-1G", "999999999P", "8191P", "8192P", "16383T", "1GG", "1BK", "123456789", "1234567890", "1kB", "G1"};
  for(const string& s : mems) {
    cout << "readMem " << q(s);
    try { cout << " " << Global::readMem(s); }
    catch(const StringError& e) { cout << " throws " << q(e.what()); }
    cout << "\n";
  }
  // roundStatic
  double xs[] = {0.0, 1.0, -1.0, 0.5, -0.5, 1.25, 2.675, -2.675, 123.456789, 1e-9, 3.14159265};
  double scales[] = {1.0, 10.0, 100.0, 1000.0, 0.1, 4.0};
  for(double x : xs)
    for(double sc : scales)
      cout << "roundStatic " << Global::doubleToStringHighPrecision(x) << " " << sc << " " << Global::doubleToStringHighPrecision(Global::roundStatic(x, sc)) << "\n";
  // contains / indexOf / map helpers
  vector<string> vs = {"a", "bb", "", "a"};
  cout << "contains_vec " << contains(vs, "a") << contains(vs, "") << contains(vs, "c") << contains(vs, string("bb")) << "\n";
  cout << "indexOf " << indexOf(vs, "a") << " " << indexOf(vs, "") << " " << (indexOf(vs, "zz") == string::npos) << "\n";
  vector<int> vi = {3, 1, 4, 1, 5};
  cout << "contains_vec_int " << contains(vi, 1) << contains(vi, 2) << " indexOf " << indexOf(vi, 1) << " " << indexOf(vi, 5) << "\n";
  set<string> ss = {"x", "y"};
  cout << "contains_set " << contains(ss, "x") << contains(ss, "z") << contains(ss, string("y")) << "\n";
  map<string,int> ms = {{"k", 7}, {"j", 8}};
  cout << "contains_map " << contains(ms, "k") << contains(ms, "q") << contains(ms, string("j")) << "\n";
  cout << "map_get " << map_get(ms, "k") << " " << map_get(ms, string("j")) << " " << map_get_defaulting(ms, string("q"), -1) << " " << map_get_defaulting(ms, string("k"), -1) << "\n";
  cout << "containsChar " << contains("abc", 'b') << contains("abc", 'z') << contains(string("abc"), 'c') << contains("abc", '\0') << "\n";

  // NUL bytes: `const char*` parameters end at the first NUL, std::string ones do not. Results as hex bytes.
  cout << "nul_contains " << contains(string("a\0b",3),'b') << contains(string("a\0b",3),'a') << contains(string("a\0b",3),'\0') << contains("a\0b",'b') << "\n";
  cout << "nul_allowed " << Global::stringCharsAllAllowed("b","a\0b") << Global::stringCharsAllAllowed("a","a\0b") << Global::stringCharsAllAllowed(string("a\0",2),"a") << Global::stringCharsAllAllowed(string("a\0b",3),"ab") << "\n";
  {
    const char* arr[] = {"a\0b", "c"};
    string c = Global::concat(arr, 2, ",");
    cout << "nul_concat_len " << c.size() << " " << hx(c) << "\n";
    c = Global::concat(arr, 2, ";\0;");
    cout << "nul_concat_len_delim " << c.size() << " " << hx(c) << "\n";
    vector<string> v = {string("x\0y",3), "z"};
    c = Global::concat(v, "+\0-");
    cout << "nul_concat " << c.size() << " " << hx(c) << "\n";
    c = Global::concat(v, "+\0-", 0, 2);
    cout << "nul_concat_start_end " << c.size() << " " << hx(c) << "\n";
    set<string> st = {"p", "q"};
    c = Global::concat(st, "+\0-");
    cout << "nul_concat_set " << c.size() << " " << hx(c) << "\n";
  }
  {
    string t = Global::trim(string("x\0 ",3), " \0");
    cout << "nul_trim " << t.size() << " " << hx(t) << "\n";
    t = Global::trim(string(" \0x\0 ",5), " \0x");
    cout << "nul_trim2 " << t.size() << " " << hx(t) << "\n";
    t = Global::trim(string("\0x\0",3));
    cout << "nul_trim3 " << t.size() << " " << hx(t) << "\n";
  }
  cout << "nul_map_get " << map_get(ms, "k\0x") << "\n";
  {
    string t = Global::strprintf("a\0%d", 5);
    cout << "nul_strprintf " << t.size() << " " << hx(t) << "\n";
  }
  // Non-ASCII bytes: trim and chopPrefix/chopSuffix work on bytes and may split a UTF-8 character.
  // Results that are not valid UTF-8 are printed as "lossy <name> <hex>" (the Rust String holds
  // U+FFFD there, see Global::trim); the rest as "bytes <name> <hex>".
  vector<pair<string,string>> trims = {
    {"\xC3\xA9", "\xC2\xA9"},          // "é", "©": cuts the shared A9 byte -> "\xC3"
    {"  \xC3\xA9  ", " \t\r\n\v\f"},
    {"\xC3\xA9", "\xC3\xA9"},
    {"a\xC3\xA9", "\xC2\xA9"},
    {"\xC3\xA9" "a" "\xC3\xA9", "\xC3\xA9"},
    {"\xC3\xA9" "a", "\xC3"},
    {"\xE2\x82\xAC", "\xE2\xAC"},     // "€"
    {"\xE2\x82\xAC", "\xC2\xA9"},
  };
  for(size_t i = 0; i < trims.size(); i++) {
    string t = Global::trim(trims[i].first, trims[i].second.c_str());
    cout << (validUtf8(t) ? "bytes" : "lossy") << " trim" << i << " " << t.size() << " " << hx(t) << "\n";
  }
  {
    string t = Global::chopPrefix("\xC3\xA9x", "\xC3\xA9");
    cout << (validUtf8(t) ? "bytes" : "lossy") << " chopPrefix " << t.size() << " " << hx(t) << "\n";
    t = Global::chopSuffix("x\xC3\xA9", "\xC3\xA9");
    cout << (validUtf8(t) ? "bytes" : "lossy") << " chopSuffix " << t.size() << " " << hx(t) << "\n";
  }
  return 0;
}
