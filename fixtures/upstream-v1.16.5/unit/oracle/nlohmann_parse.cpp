// Oracle for port::nlohmann::parse (katago-rs): nlohmann::json::parse and sax_parse from the
// vendored nlohmann/json 3.8.0 (upstream/KataGo/cpp/external/nlohmann_json/json.hpp).
// Reads nlohmann_parse.input.txt (one input per line, escapes \n \t \r \0 \\ \xHH) and prints,
// per input, two lines:
//   D <OK|ERR> <dump() or e.what()>          from json::parse(s)  (the DOM path)
//   S <events>                               from json::sax_parse(s, &rec)  (token classes)
// with the same escaping (bytes < 0x20 or >= 0x7F as \xHH, backslash as \\).
// Build and run (from the repo root, output outside the repo; upstream untouched):
//   g++ -std=c++14 -O1 -I upstream/KataGo/cpp/external -o /tmp/nlohmann_parse \
//     fixtures/upstream-v1.16.5/unit/oracle/nlohmann_parse.cpp
//   /tmp/nlohmann_parse < fixtures/upstream-v1.16.5/unit/oracle/nlohmann_parse.input.txt \
//     > fixtures/upstream-v1.16.5/unit/oracle/nlohmann_parse.txt
// Generated with g++ 16.2.1 / glibc 2.44 on x86_64 Linux.
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <string>
#include "nlohmann_json/json.hpp"
using json = nlohmann::json;
using namespace std;

static int hexv(char c) { return (c >= '0' && c <= '9') ? c - '0' : (c >= 'a' && c <= 'f') ? c - 'a' + 10 : c - 'A' + 10; }
static string unesc(const string& s) {
  string o;
  for(size_t i = 0; i < s.size(); i++) {
    if(s[i] == '\\' && i + 1 < s.size()) {
      char c = s[++i];
      if(c == 'n') o += '\n';
      else if(c == 't') o += '\t';
      else if(c == 'r') o += '\r';
      else if(c == '0') o += '\0';
      else if(c == 'x') { o += (char)(hexv(s[i+1]) * 16 + hexv(s[i+2])); i += 2; }
      else o += c;
    }
    else o += s[i];
  }
  return o;
}
static string esc(const string& s) {
  string o;
  for(unsigned char c : s) {
    if(c == '\\') o += "\\\\";
    else if(c < 0x20 || c >= 0x7F) { char b[8]; snprintf(b, 8, "\\x%02x", c); o += b; }
    else o += (char)c;
  }
  return o;
}

struct Rec {
  std::string ev;
  void put(const std::string& s) { if(!ev.empty()) ev += ' '; ev += s; }
  bool null() { put("n"); return true; }
  bool boolean(bool b) { put(b ? "b1" : "b0"); return true; }
  bool number_integer(json::number_integer_t v) { put("i" + to_string(v)); return true; }
  bool number_unsigned(json::number_unsigned_t v) { put("u" + to_string(v)); return true; }
  bool number_float(json::number_float_t v, const std::string& s) {
    uint64_t b; memcpy(&b, &v, 8); char buf[32]; snprintf(buf, 32, "%016llX", (unsigned long long)b);
    put(std::string("f") + buf + "/" + esc(s)); return true;
  }
  bool string(json::string_t& s) { put("s" + esc(s)); return true; }
  bool binary(json::binary_t&) { put("B"); return true; }
  bool start_object(size_t) { put("{"); return true; }
  bool key(json::string_t& s) { put("k" + esc(s)); return true; }
  bool end_object() { put("}"); return true; }
  bool start_array(size_t) { put("["); return true; }
  bool end_array() { put("]"); return true; }
  bool parse_error(size_t pos, const std::string& tok, const nlohmann::detail::exception& e) {
    put("E" + to_string(pos) + "|" + esc(tok) + "|" + esc(e.what())); return false;
  }
};

int main() {
  std::string line;
  while(getline(cin, line)) {
    std::string in = unesc(line);
    try {
      json j = json::parse(in);
      cout << "D OK " << esc(j.dump()) << "\n";
    }
    catch(nlohmann::detail::exception& e) {
      cout << "D ERR " << esc(e.what()) << "\n";
    }
    Rec r;
    bool ok = json::sax_parse(in, &r);
    cout << "S " << (ok ? "1 " : "0 ") << r.ev << "\n";
  }
  return 0;
}
