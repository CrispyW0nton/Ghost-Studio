#include "Resources_Assets/ResourceManager.h"

#include <cctype>
#include <string>
#include <string_view>

namespace ghostrigger::core::assets::core::assets::resource_manager {
namespace {

std::string lower_ascii(std::string_view value) {
    std::string result;
    result.reserve(value.size());
    for (const unsigned char character : value) {
        result.push_back(static_cast<char>(std::tolower(character)));
    }
    return result;
}

std::string strip_ascii_whitespace(std::string_view value) {
    std::size_t start = 0;
    std::size_t end = value.size();
    while (start < end && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return std::string(value.substr(start, end - start));
}

std::string json_string(std::string_view value) {
    std::string result = "\"";
    for (const char character : value) {
        switch (character) {
        case '\\':
            result.append("\\\\");
            break;
        case '"':
            result.append("\\\"");
            break;
        case '\b':
            result.append("\\b");
            break;
        case '\f':
            result.append("\\f");
            break;
        case '\n':
            result.append("\\n");
            break;
        case '\r':
            result.append("\\r");
            break;
        case '\t':
            result.append("\\t");
            break;
        default:
            result.push_back(character);
            break;
        }
    }
    result.push_back('"');
    return result;
}

std::string normalize_extension(std::string_view extension) {
    std::string cleaned = lower_ascii(strip_ascii_whitespace(extension));
    if (!cleaned.empty() && cleaned.front() == '.') {
        cleaned.erase(cleaned.begin());
    }
    return cleaned;
}

} // namespace

std::string resource_key(std::string_view name, int resource_type) {
    std::string key = lower_ascii(name);
    key.push_back(':');
    key.append(std::to_string(resource_type));
    return key;
}

std::string texture_name_candidates_json(std::string_view name) {
    const std::string key = lower_ascii(strip_ascii_whitespace(name));
    if (key.empty()) {
        return "[]";
    }

    std::string result = "[";
    result.append(json_string(key));
    if (key == "c_drex01") {
        result.append(",");
        result.append(json_string("c_drexl01"));
    }
    if (key == "x_rptnblstr_001") {
        result.append(",");
        result.append(json_string("w_rptnblstr_001"));
    }
    result.push_back(']');
    return result;
}

int extension_to_resource_type(std::string_view extension) noexcept {
    const std::string ext = normalize_extension(extension);
    if (ext == "bmp") {
        return 1;
    }
    if (ext == "tga") {
        return 3;
    }
    if (ext == "wav" || ext == "mp3") {
        return 4;
    }
    if (ext == "plt") {
        return 6;
    }
    if (ext == "ini") {
        return 7;
    }
    if (ext == "txt") {
        return 10;
    }
    if (ext == "mdl") {
        return 2002;
    }
    if (ext == "nss") {
        return 2009;
    }
    if (ext == "ncs") {
        return 2010;
    }
    if (ext == "are") {
        return 2012;
    }
    if (ext == "ifo") {
        return 2013;
    }
    if (ext == "txi") {
        return 2014;
    }
    if (ext == "git") {
        return 2015;
    }
    if (ext == "wok") {
        return 2016;
    }
    if (ext == "2da") {
        return 2017;
    }
    if (ext == "utc") {
        return 2023;
    }
    if (ext == "dlg") {
        return 2029;
    }
    if (ext == "utd") {
        return 2038;
    }
    if (ext == "utp") {
        return 2044;
    }
    if (ext == "lyt") {
        return 3000;
    }
    if (ext == "vis") {
        return 3001;
    }
    if (ext == "tpc") {
        return 3007;
    }
    if (ext == "mdx") {
        return 3008;
    }
    if (ext == "mod") {
        return 3011;
    }
    return -1;
}

const char* resource_type_to_extension(int resource_type) noexcept {
    switch (resource_type) {
    case 1:
        return "bmp";
    case 3:
        return "tga";
    case 4:
        return "mp3";
    case 6:
        return "plt";
    case 7:
        return "ini";
    case 10:
        return "txt";
    case 2002:
        return "mdl";
    case 2009:
        return "nss";
    case 2010:
        return "ncs";
    case 2012:
        return "are";
    case 2013:
        return "ifo";
    case 2014:
        return "txi";
    case 2015:
        return "git";
    case 2016:
        return "wok";
    case 2017:
        return "2da";
    case 2023:
        return "utc";
    case 2029:
        return "dlg";
    case 2038:
        return "utd";
    case 2044:
        return "utp";
    case 3000:
        return "lyt";
    case 3001:
        return "vis";
    case 3007:
        return "tpc";
    case 3008:
        return "mdx";
    case 3011:
        return "mod";
    default:
        return "";
    }
}

const char* resource_manager_schema_json() noexcept {
    static constexpr const char* kJson =
        R"({"schema":"resource_manager_native.v1",)"
        R"("source":"src/core/assets/resource_manager.py",)"
        R"("native_scope":["_key","_texture_name_candidates","EXT_TO_TYPE","TYPE_TO_EXT"],)"
        R"("python_fallback":["BIF/ERF archive indexing","lazy file reads","game install discovery","TPC decoding","texture image alpha fixes","model texture audit orchestration"],)"
        R"("reason_python_fallback":"archive IO, install configuration, image decoding, and model-object texture resolution still depend on Python services and PyKotor/Pillow-facing objects"})";
    return kJson;
}

} // namespace ghostrigger::core::assets::core::assets::resource_manager
