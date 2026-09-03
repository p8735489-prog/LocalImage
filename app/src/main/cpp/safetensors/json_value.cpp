#include "json_value.h"

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace localimage::json {
namespace {

class Parser {
public:
    Parser(const char* data, size_t size) : begin_(data), p_(data), end_(data + size) {}

    bool run(Value& out, std::string& error) {
        skipWs();
        if (!parseValue(out, error)) return false;
        skipWs();
        if (p_ != end_) return fail("trailing data", error);
        return true;
    }

private:
    const char* begin_;
    const char* p_;
    const char* end_;

    bool fail(const char* message, std::string& error) {
        error = std::string(message) + " at byte " + std::to_string(static_cast<size_t>(p_ - begin_));
        return false;
    }

    void skipWs() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_;
    }

    bool consume(char c) {
        if (p_ < end_ && *p_ == c) { ++p_; return true; }
        return false;
    }

    bool parseValue(Value& out, std::string& error) {
        if (p_ == end_) return fail("unexpected end", error);
        switch (*p_) {
            case '{': return parseObject(out, error);
            case '[': return parseArray(out, error);
            case '"': { std::string s; if (!parseString(s, error)) return false; out = Value(s); return true; }
            case 't': return parseLiteral("true", Value(true), out, error);
            case 'f': return parseLiteral("false", Value(false), out, error);
            case 'n': return parseLiteral("null", Value(), out, error);
            default: return parseNumber(out, error);
        }
    }

    bool parseLiteral(const char* literal, const Value& value, Value& out, std::string& error) {
        const size_t n = std::char_traits<char>::length(literal);
        if (static_cast<size_t>(end_ - p_) < n || std::string(p_, p_ + n) != literal) return fail("invalid literal", error);
        p_ += n; out = value; return true;
    }

    static bool hex(char c, uint32_t& v) {
        if (c >= '0' && c <= '9') { v = c - '0'; return true; }
        if (c >= 'a' && c <= 'f') { v = c - 'a' + 10; return true; }
        if (c >= 'A' && c <= 'F') { v = c - 'A' + 10; return true; }
        return false;
    }

    bool appendUtf8(uint32_t cp, std::string& out, std::string& error) {
        if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return fail("invalid unicode code point", error);
        if (cp <= 0x7F) out.push_back(static_cast<char>(cp));
        else if (cp <= 0x7FF) { out.push_back(static_cast<char>(0xC0 | (cp >> 6))); out.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
        else if (cp <= 0xFFFF) { out.push_back(static_cast<char>(0xE0 | (cp >> 12))); out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F))); out.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
        else { out.push_back(static_cast<char>(0xF0 | (cp >> 18))); out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F))); out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F))); out.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
        return true;
    }

    bool parseString(std::string& out, std::string& error) {
        if (!consume('"')) return fail("expected string", error);
        out.clear();
        while (p_ < end_) {
            unsigned char c = static_cast<unsigned char>(*p_++);
            if (c == '"') return true;
            if (c < 0x20) return fail("control character in string", error);
            if (c != '\\') { out.push_back(static_cast<char>(c)); continue; }
            if (p_ == end_) return fail("unterminated escape", error);
            const char e = *p_++;
            switch (e) {
                case '"': out.push_back('"'); break; case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break; case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break; case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break; case 't': out.push_back('\t'); break;
                case 'u': {
                    if (end_ - p_ < 4) return fail("short unicode escape", error);
                    uint32_t cp = 0, digit;
                    for (int i = 0; i < 4; ++i) { if (!hex(p_[i], digit)) return fail("invalid unicode escape", error); cp = (cp << 4) | digit; }
                    p_ += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (end_ - p_ < 6 || p_[0] != '\\' || p_[1] != 'u') return fail("missing unicode surrogate", error);
                        uint32_t low = 0;
                        for (int i = 0; i < 4; ++i) { if (!hex(p_[2 + i], digit)) return fail("invalid low surrogate", error); low = (low << 4) | digit; }
                        if (low < 0xDC00 || low > 0xDFFF) return fail("invalid low surrogate", error);
                        p_ += 6; cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) return fail("unexpected low surrogate", error);
                    if (!appendUtf8(cp, out, error)) return false;
                    break;
                }
                default: return fail("invalid string escape", error);
            }
        }
        return fail("unterminated string", error);
    }

    bool parseNumber(Value& out, std::string& error) {
        const char* start = p_;
        if (p_ < end_ && *p_ == '-') ++p_;
        if (p_ == end_) return fail("invalid number", error);
        if (*p_ == '0') ++p_;
        else {
            if (*p_ < '1' || *p_ > '9') return fail("invalid number", error);
            while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
        }
        bool floating = false;
        if (p_ < end_ && *p_ == '.') { floating = true; ++p_; if (p_ == end_ || *p_ < '0' || *p_ > '9') return fail("invalid fraction", error); while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_; }
        if (p_ < end_ && (*p_ == 'e' || *p_ == 'E')) { floating = true; ++p_; if (p_ < end_ && (*p_ == '+' || *p_ == '-')) ++p_; if (p_ == end_ || *p_ < '0' || *p_ > '9') return fail("invalid exponent", error); while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_; }
        std::string s(start, p_);
        if (!floating) {
            if (!s.empty() && s[0] == '-') {
                int64_t v; auto r = std::from_chars(s.data(), s.data() + s.size(), v); if (r.ec == std::errc() && r.ptr == s.data() + s.size()) { out = Value(v); return true; }
            } else {
                uint64_t v; auto r = std::from_chars(s.data(), s.data() + s.size(), v); if (r.ec == std::errc() && r.ptr == s.data() + s.size()) { out = Value(v); return true; }
            }
            return fail("integer overflow", error);
        }
        char* ep = nullptr; errno = 0; double d = std::strtod(s.c_str(), &ep);
        if (errno == ERANGE || !std::isfinite(d) || ep != s.c_str() + s.size()) return fail("invalid floating number", error);
        out = Value(d); return true;
    }

    bool parseArray(Value& out, std::string& error) {
        if (!consume('[')) return fail("expected array", error);
        Value::Array a; skipWs();
        if (consume(']')) { out = Value(std::move(a)); return true; }
        while (true) {
            skipWs(); Value v; if (!parseValue(v, error)) return false; a.push_back(std::move(v)); skipWs();
            if (consume(']')) break;
            if (!consume(',')) return fail("expected comma in array", error);
        }
        out = Value(std::move(a)); return true;
    }

    bool parseObject(Value& out, std::string& error) {
        if (!consume('{')) return fail("expected object", error);
        Value::Object o; skipWs();
        if (consume('}')) { out = Value(std::move(o)); return true; }
        while (true) {
            skipWs(); std::string key; if (!parseString(key, error)) return false; skipWs(); if (!consume(':')) return fail("expected colon in object", error); skipWs();
            Value v; if (!parseValue(v, error)) return false;
            if (o.find(key) != o.end()) return fail("duplicate object key", error);
            o.emplace(std::move(key), std::move(v)); skipWs();
            if (consume('}')) break;
            if (!consume(',')) return fail("expected comma in object", error);
        }
        out = Value(std::move(o)); return true;
    }
};

} // namespace

bool Value::getUnsigned(uint64_t& out) const {
    if (const auto* v = std::get_if<uint64_t>(&value_)) { out = *v; return true; }
    if (const auto* v = std::get_if<int64_t>(&value_)) { if (*v >= 0) { out = static_cast<uint64_t>(*v); return true; } }
    return false;
}

bool Value::getString(std::string& out) const { if (const auto* s = string()) { out = *s; return true; } return false; }

bool parse(const char* data, size_t size, Value& out, std::string& error) {
    if (!data && size != 0) { error = "null JSON buffer"; return false; }
    Parser p(data, size); return p.run(out, error);
}

} // namespace localimage::json
