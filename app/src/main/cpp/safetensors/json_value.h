#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace localimage::json {

class Value {
public:
    using Object = std::map<std::string, Value>;
    using Array = std::vector<Value>;
    using Storage = std::variant<std::nullptr_t, bool, int64_t, uint64_t, double, std::string, Array, Object>;

    Value() : value_(nullptr) {}
    explicit Value(Storage value) : value_(std::move(value)) {}

    bool isNull() const { return std::holds_alternative<std::nullptr_t>(value_); }
    bool isBool() const { return std::holds_alternative<bool>(value_); }
    bool isInteger() const { return std::holds_alternative<int64_t>(value_) || std::holds_alternative<uint64_t>(value_); }
    bool isNumber() const { return isInteger() || std::holds_alternative<double>(value_); }
    bool isString() const { return std::holds_alternative<std::string>(value_); }
    bool isArray() const { return std::holds_alternative<Array>(value_); }
    bool isObject() const { return std::holds_alternative<Object>(value_); }

    const std::string* string() const { return std::get_if<std::string>(&value_); }
    const Array* array() const { return std::get_if<Array>(&value_); }
    const Object* object() const { return std::get_if<Object>(&value_); }

    bool getUnsigned(uint64_t& out) const;
    bool getString(std::string& out) const;

private:
    Storage value_;
};

bool parse(const char* data, size_t size, Value& out, std::string& error);

} // namespace localimage::json
