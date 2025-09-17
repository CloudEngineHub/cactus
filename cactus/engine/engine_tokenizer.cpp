#include "engine.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

namespace cactus {
namespace engine {

BPETokenizer::BPETokenizer() 
    : vocab_size_(0), unk_token_id_(0), bos_token_id_(1), eos_token_id_(2),
      vocab_mmap_ptr_(nullptr), vocab_mmap_size_(0),
      merges_mmap_ptr_(nullptr), merges_mmap_size_(0),
      has_chat_template_(false) {
}

BPETokenizer::~BPETokenizer() {
    cleanup_mmap();
}

void BPETokenizer::cleanup_mmap() {
    if (vocab_mmap_ptr_ && vocab_mmap_ptr_ != MAP_FAILED) {
        munmap(vocab_mmap_ptr_, vocab_mmap_size_);
        vocab_mmap_ptr_ = nullptr;
    }
    if (merges_mmap_ptr_ && merges_mmap_ptr_ != MAP_FAILED) {
        munmap(merges_mmap_ptr_, merges_mmap_size_);
        merges_mmap_ptr_ = nullptr;
    }
}

bool BPETokenizer::load_vocabulary_mmap(const std::string& vocab_file, const std::string& merges_file) {
    int vocab_fd = open(vocab_file.c_str(), O_RDONLY);
    if (vocab_fd == -1) return false;
    
    struct stat vocab_stat;
    if (fstat(vocab_fd, &vocab_stat) == -1) {
        close(vocab_fd);
        return false;
    }
    
    vocab_mmap_size_ = vocab_stat.st_size;
    vocab_mmap_ptr_ = mmap(nullptr, vocab_mmap_size_, PROT_READ, MAP_PRIVATE, vocab_fd, 0);
    close(vocab_fd);
    
    if (vocab_mmap_ptr_ == MAP_FAILED) return false;
    
    std::string vocab_content(static_cast<char*>(vocab_mmap_ptr_), vocab_mmap_size_);
    std::istringstream vocab_stream(vocab_content);
    
    std::string line;
    uint32_t id = 0;
    token_to_id_.clear();
    id_to_token_.clear();
    
    while (std::getline(vocab_stream, line)) {
        if (line.empty()) continue;
        token_to_id_[line] = id;
        id_to_token_.push_back(line);
        id++;
    }
    vocab_size_ = id;
    
    int merges_fd = open(merges_file.c_str(), O_RDONLY);
    if (merges_fd == -1) return false;
    
    struct stat merges_stat;
    if (fstat(merges_fd, &merges_stat) == -1) {
        close(merges_fd);
        return false;
    }
    
    merges_mmap_size_ = merges_stat.st_size;
    merges_mmap_ptr_ = mmap(nullptr, merges_mmap_size_, PROT_READ, MAP_PRIVATE, merges_fd, 0);
    close(merges_fd);
    
    if (merges_mmap_ptr_ == MAP_FAILED) return false;
    
    std::string merges_content(static_cast<char*>(merges_mmap_ptr_), merges_mmap_size_);
    std::istringstream merges_stream(merges_content);
    
    merge_rules_.clear();
    uint32_t priority = 0;
    
    while (std::getline(merges_stream, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string first, second;
        if (iss >> first >> second) {
            std::string merged = first + second;
            merge_rules_.emplace_back(first, second, merged, priority);
            
            std::string key = first + "\x00" + second; 
            auto it = merge_map_.find(key);
            if (it == merge_map_.end() || priority < it->second) {
                merge_map_[key] = priority;
            }
            priority++;
        }
    }
    
    std::sort(merge_rules_.begin(), merge_rules_.end(),
              [](const MergeRule& a, const MergeRule& b) {
                  return a.priority < b.priority;
              });
    
    return true;
}

bool BPETokenizer::load_vocabulary_with_config(const std::string& vocab_file, const std::string& merges_file, const std::string& config_file) {
    if (!load_vocabulary_mmap(vocab_file, merges_file)) {
        return false;
    }
    
    std::ifstream config_stream(config_file);
    if (!config_stream.is_open()) {
        return true;
    }
    
    std::string line;
    while (std::getline(config_stream, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);
        
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        
        if (key == "eos_token_id") {
            eos_token_id_ = std::stoul(value);
        } else if (key == "pad_token_id") {
            if (unk_token_id_ == 0) {
                unk_token_id_ = std::stoul(value);
            }
        } else if (key == "unk_token_id" && value != "null") {
            unk_token_id_ = std::stoul(value);
        } else if (key == "bos_token_id" && value != "null") {
            bos_token_id_ = std::stoul(value);
        } else if (key == "vocab_size") {
            if (std::stoul(value) != vocab_size_) {
            }
        }
    }
    
    std::string special_tokens_path = config_file.substr(0, config_file.find_last_of("/\\")) + "/special_tokens.json";
    load_special_tokens(special_tokens_path);
    
    std::string template_path = config_file.substr(0, config_file.find_last_of("/\\")) + "/chat_template.jinja2";
    load_chat_template(template_path);
    
    return true;
}

void BPETokenizer::load_special_tokens(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        return;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    
    size_t pos = content.find("\"special_tokens\"");
    if (pos == std::string::npos) return;
    
    pos = content.find("{", pos);
    if (pos == std::string::npos) return;
    
    size_t end_pos = content.find("}", pos);
    if (end_pos == std::string::npos) return;
    
    std::string special_tokens_section = content.substr(pos + 1, end_pos - pos - 1);
    
    std::istringstream iss(special_tokens_section);
    std::string line;
    
    while (std::getline(iss, line)) {
        size_t colon_pos = line.find(":");
        if (colon_pos == std::string::npos) continue;
        
        std::string id_part = line.substr(0, colon_pos);
        std::string token_part = line.substr(colon_pos + 1);
        
        size_t id_start = id_part.find("\"");
        size_t id_end = id_part.find("\"", id_start + 1);
        if (id_start == std::string::npos || id_end == std::string::npos) continue;
        
        std::string id_str = id_part.substr(id_start + 1, id_end - id_start - 1);
        uint32_t token_id = std::stoul(id_str);
        
        size_t token_start = token_part.find("\"");
        size_t token_end = token_part.rfind("\"");
        if (token_start == std::string::npos || token_end == std::string::npos || token_start >= token_end) continue;
        
        std::string token_content = token_part.substr(token_start + 1, token_end - token_start - 1);
        
        special_tokens_[token_content] = token_id;
    }
    
}

std::vector<std::string> BPETokenizer::split_with_special_tokens(const std::string& text) const {
    std::vector<std::string> result;
    
    size_t start = 0;
    while (start < text.size()) {
        size_t best_match_pos = text.size();
        size_t best_match_len = 0;
        std::string best_special_token;
        
        for (const auto& [special_token, token_id] : special_tokens_) {
            size_t pos = text.find(special_token, start);
            if (pos != std::string::npos && pos < best_match_pos) {
                best_match_pos = pos;
                best_match_len = special_token.length();
                best_special_token = special_token;
            }
        }
        
        if (best_match_pos < text.size()) {
            if (best_match_pos > start) {
                std::string before = text.substr(start, best_match_pos - start);
                result.push_back(before);
            }
            
            result.push_back(best_special_token);
            start = best_match_pos + best_match_len;
        } else {
            if (start < text.size()) {
                result.push_back(text.substr(start));
            }
            break;
        }
    }
    
    return result;
}

void BPETokenizer::init_byte_mappings() const {
    if (!byte_to_unicode_.empty()) return;
    
    std::vector<int> bytes;
    
    for (int i = 33; i <= 126; ++i) {
        bytes.push_back(i);
    }
    
  
    for (int i = 161; i <= 255; ++i) {
        bytes.push_back(i);
    }
    
    std::vector<int> remaining_bytes;
    for (int i = 0; i <= 32; ++i) remaining_bytes.push_back(i);
    remaining_bytes.push_back(127);
    for (int i = 128; i <= 160; ++i) remaining_bytes.push_back(i);
    
    int unicode_start = 256;
    for (int byte : remaining_bytes) {
        bytes.push_back(byte);
    }
    
    for (size_t i = 0; i < bytes.size(); ++i) {
        uint8_t byte = static_cast<uint8_t>(bytes[i]);
        
        if (byte >= 33 && byte <= 126) {
            std::string unicode_char(1, static_cast<char>(byte));
            byte_to_unicode_[byte] = unicode_char;
            unicode_to_byte_[unicode_char] = byte;
        } else if (byte >= 161 && byte <= 255) {
            std::string unicode_char;
            unicode_char += static_cast<char>(0xC0 | (byte >> 6));
            unicode_char += static_cast<char>(0x80 | (byte & 0x3F));
            byte_to_unicode_[byte] = unicode_char;
            unicode_to_byte_[unicode_char] = byte;
        } else {
            int unicode_point = unicode_start++;
            std::string unicode_char;
            if (unicode_point < 0x800) {
                unicode_char += static_cast<char>(0xC0 | (unicode_point >> 6));
                unicode_char += static_cast<char>(0x80 | (unicode_point & 0x3F));
            } else {
                unicode_char += static_cast<char>(0xE0 | (unicode_point >> 12));
                unicode_char += static_cast<char>(0x80 | ((unicode_point >> 6) & 0x3F));
                unicode_char += static_cast<char>(0x80 | (unicode_point & 0x3F));
            }
            byte_to_unicode_[byte] = unicode_char;
            unicode_to_byte_[unicode_char] = byte;
        }
    }
}

std::string BPETokenizer::bytes_to_unicode(const std::string& text) const {
    init_byte_mappings();
    
    std::string result;
    for (uint8_t byte : text) {
        result += byte_to_unicode_.at(byte);
    }
    return result;
}

std::string BPETokenizer::unicode_to_bytes(const std::string& text) const {
    init_byte_mappings();
    
    std::string result;
    size_t i = 0;
    while (i < text.length()) {
        std::string unicode_char;
        
        if ((text[i] & 0x80) == 0) {
            unicode_char = text.substr(i, 1);
            i += 1;
        } else if ((text[i] & 0xE0) == 0xC0) {
            unicode_char = text.substr(i, 2);
            i += 2;
        } else if ((text[i] & 0xF0) == 0xE0) {
            unicode_char = text.substr(i, 3);
            i += 3;
        } else {
            unicode_char = text.substr(i, 1);
            i += 1;
        }
        
        auto it = unicode_to_byte_.find(unicode_char);
        if (it != unicode_to_byte_.end()) {
            result += static_cast<char>(it->second);
        } else {
            result += '?';
        }
    }
    return result;
}

std::vector<std::string> BPETokenizer::byte_level_split(const std::string& text) const {
    std::string unicode_text = bytes_to_unicode(text);
    
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < unicode_text.length()) {
        size_t char_len = 1;
        
        if ((unicode_text[i] & 0x80) == 0) {
            char_len = 1;
        } else if ((unicode_text[i] & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((unicode_text[i] & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((unicode_text[i] & 0xF8) == 0xF0) {
            char_len = 4;
        }
        
        if (i + char_len <= unicode_text.length()) {
            chars.push_back(unicode_text.substr(i, char_len));
        }
        i += char_len;
    }
    
    return chars;
}


std::pair<int, uint32_t> BPETokenizer::find_best_merge_fast(const std::vector<std::string>& tokens) const {
    int best_pos = -1;
    uint32_t best_priority = UINT32_MAX;
    
    // Look for the best merge in a single pass using hash map
    for (size_t i = 0; i < tokens.size() - 1; ++i) {
        std::string key = tokens[i] + "\x00" + tokens[i + 1];
        auto it = merge_map_.find(key);
        if (it != merge_map_.end()) {
            if (it->second < best_priority) {
                best_priority = it->second;
                best_pos = static_cast<int>(i);
            }
        }
    }
    
    return {best_pos, best_priority};
}

std::vector<std::string> BPETokenizer::apply_bpe(const std::vector<std::string>& tokens) const {
    if (tokens.size() <= 1) return tokens;
    
    std::vector<std::string> current_tokens = tokens;
    
    
    while (true) {
        auto [merge_pos, priority] = find_best_merge_fast(current_tokens);
        if (merge_pos == -1) break;
        
        
        std::vector<std::string> new_tokens;
        new_tokens.reserve(current_tokens.size() - 1);  // Pre-allocate
        
        for (int i = 0; i < static_cast<int>(current_tokens.size()); ++i) {
            if (i == merge_pos) {
                std::string merged = current_tokens[i] + current_tokens[i + 1];
                new_tokens.push_back(merged);
                i++;  // Skip next token
            } else {
                new_tokens.push_back(current_tokens[i]);
            }
        }
        current_tokens = std::move(new_tokens);
    }
    
    return current_tokens;
}

std::vector<uint32_t> BPETokenizer::encode(const std::string& text) const {
    if (text.empty()) return {};
    
    
    auto text_segments = split_with_special_tokens(text);
    
    
    std::vector<uint32_t> token_ids;
    
    for (const auto& segment : text_segments) {
        auto special_it = special_tokens_.find(segment);
        if (special_it != special_tokens_.end()) {
            token_ids.push_back(special_it->second);
        } else {
            auto chars = byte_level_split(segment);
            auto bpe_tokens = apply_bpe(chars);
            
            
            for (const auto& token : bpe_tokens) {
                auto it = token_to_id_.find(token);
                if (it != token_to_id_.end()) {
                    token_ids.push_back(it->second);
                } else {
                    token_ids.push_back(unk_token_id_);
                }
            }
        }
    }
    
    return token_ids;
}

std::string BPETokenizer::decode(const std::vector<uint32_t>& tokens) const {
    std::string unicode_result;
    for (uint32_t token_id : tokens) {
        if (token_id < id_to_token_.size()) {
            unicode_result += id_to_token_[token_id];
        }
    }
    
    std::string result = unicode_to_bytes(unicode_result);
    
    return result;
}

void BPETokenizer::load_chat_template(const std::string& template_file) {
    std::ifstream file(template_file);
    if (!file.is_open()) {
        has_chat_template_ = false;
        return;
    }
    
    chat_template_ = std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    has_chat_template_ = !chat_template_.empty();
}

std::string BPETokenizer::apply_template_substitutions(const std::string& template_str, const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const {
    
    if (!tools_json.empty()) {
        std::string result;
        
        result += "<|im_start|>system\n";
        
        bool has_system_msg = false;
        for (const auto& msg : messages) {
            if (msg.role == "system") {
                result += msg.content;
                result += "\n\n";
                has_system_msg = true;
                break;
            }
        }
        
        result += "You can respond normally to the user's request. If you need to call tools, respond with a JSON object containing `tool_calls`.\n";
        result += "Only call tools when they are necessary to fulfill the user's request.\n";
        result += "You can call any of the following tools to satisfy the user's requests: [\n";
        result += tools_json;
        result += "\n]\n";
        result += "Example tool call syntax:\n";
        result += "{\n";
        result += "  \"tool_calls\": [\n";
        result += "    {\n";
        result += "      \"name\": \"tool_name\",\n";
        result += "      \"arguments\": {\n";
        result += "        \"arg1\": \"some_value\"\n";
        result += "      },\n";
        result += "      \"id\": \"call_1___\"\n";
        result += "    }\n";
        result += "  ]\n";
        result += "}";
        result += "<|im_end|>\n";
        
        for (const auto& msg : messages) {
            if (msg.role == "system" && has_system_msg) {
                continue; 
            } else if (msg.role == "user") {
                result += "<|im_start|>user\n" + msg.content + "<|im_end|>\n";
            } else if (msg.role == "assistant") {
                result += "<|im_start|>assistant\n" + msg.content + "<|im_end|>\n";
            }
        }
        
        if (add_generation_prompt) {
            result += "<|im_start|>assistant\n";
        }
        
        return result;
    }
    
    std::string result = template_str;
    
    if (result.find("{% for message in messages %}") != std::string::npos) {
        std::string message_part;
        for (const auto& msg : messages) {
            std::string role = msg.role;
            std::string content = msg.content;
            
            if (role == "system") {
                message_part += "<|im_start|>system\n" + content + "<|im_end|>\n";
            } else if (role == "user") {
                message_part += "<|im_start|>user\n" + content + "<|im_end|>\n";
            } else if (role == "assistant") {
                message_part += "<|im_start|>assistant\n" + content + "<|im_end|>\n";
            }
        }
        
        if (add_generation_prompt) {
            message_part += "<|im_start|>assistant\n";
        }
        
        size_t start = result.find("{% for message in messages %}");
        size_t end = result.find("{% endfor %}");
        if (start != std::string::npos && end != std::string::npos) {
            result = result.substr(0, start) + message_part + result.substr(end + 12);
        }
    } else {
        std::string formatted_messages;
        for (const auto& msg : messages) {
            if (msg.role == "system") {
                formatted_messages += "<|im_start|>system\n" + msg.content + "<|im_end|>\n";
            } else if (msg.role == "user") {
                formatted_messages += "<|im_start|>user\n" + msg.content + "<|im_end|>\n";
            } else if (msg.role == "assistant") {
                formatted_messages += "<|im_start|>assistant\n" + msg.content + "<|im_end|>\n";
            }
        }
        
        if (add_generation_prompt) {
            formatted_messages += "<|im_start|>assistant\n";
        }
        
        result = formatted_messages;
    }
    
    return result;
}

std::string BPETokenizer::format_chat_prompt(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const {
    if (has_chat_template_ && !chat_template_.empty()) {
        return apply_template_substitutions(chat_template_, messages, add_generation_prompt, tools_json);
    }
    
    std::string formatted;
    for (const auto& msg : messages) {
        if (msg.role == "system") {
            formatted += "<|im_start|>system\n" + msg.content + "<|im_end|>\n";
        } else if (msg.role == "user") {
            formatted += "<|im_start|>user\n" + msg.content + "<|im_end|>\n";
        } else if (msg.role == "assistant") {
            formatted += "<|im_start|>assistant\n" + msg.content + "<|im_end|>\n";
        }
    }
    
    if (add_generation_prompt) {
        formatted += "<|im_start|>assistant\n";
    }
    
    return formatted;
}

std::vector<uint32_t> BPETokenizer::apply_chat_template(const std::vector<ChatMessage>& messages, bool add_generation_prompt) const {
    std::string formatted_prompt = format_chat_prompt(messages, add_generation_prompt);
    return encode(formatted_prompt);
}

}
}