/**
 * @file stratum_client.cpp
 * @brief Lightweight Stratum v1 TCP client implementation for Blake2bCudaMiner.
 */

#include "stratum_client.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <fcntl.h>
#include <openssl/sha.h>

static void sha256_double(const uint8_t* data, size_t len, uint8_t out[32]) {
    uint8_t hash1[32];
    SHA256(data, len, hash1);
    SHA256(hash1, 32, out);
}

static std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byteString = hex.substr(i, 2);
        uint8_t byte = (uint8_t)strtol(byteString.c_str(), nullptr, 16);
        bytes.push_back(byte);
    }
    return bytes;
}

StratumClient::StratumClient(const std::string& host, int port, const std::string& user, const std::string& pass)
    : host_(host), port_(port), user_(user), pass_(pass), socket_fd_(-1),
      connected_(false), difficulty_(16.0), extranonce1_("00000001"),
      extranonce2_size_(4), message_id_(1) {}

StratumClient::~StratumClient() {
    disconnect_server();
}

bool StratumClient::connect_to_server() {
    socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd_ < 0) return false;

    struct hostent* server = gethostbyname(host_.c_str());
    if (!server) return false;

    struct sockaddr_in serv_addr;
    std::memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    std::memcpy(&serv_addr.sin_addr.s_addr, server->h_addr, server->h_length);
    serv_addr.sin_port = htons(port_);

    if (connect(socket_fd_, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        close(socket_fd_);
        socket_fd_ = -1;
        return false;
    }

    // Set non-blocking socket
    int flags = fcntl(socket_fd_, F_GETFL, 0);
    fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK);

    connected_ = true;

    // 1. mining.subscribe
    std::string sub_req = "{\"id\": 1, \"method\": \"mining.subscribe\", \"params\": [\"Blake2bCudaMiner/1.2\"]}\n";
    send_line(sub_req);

    // 2. mining.authorize
    std::string auth_req = "{\"id\": 2, \"method\": \"mining.authorize\", \"params\": [\"" + user_ + "\", \"" + pass_ + "\"]}\n";
    send_line(auth_req);

    return true;
}

void StratumClient::disconnect_server() {
    if (socket_fd_ >= 0) {
        close(socket_fd_);
        socket_fd_ = -1;
    }
    connected_ = false;
}

bool StratumClient::send_line(const std::string& json_str) {
    if (!connected_ || socket_fd_ < 0) return false;
    ssize_t sent = send(socket_fd_, json_str.c_str(), json_str.length(), 0);
    return sent == (ssize_t)json_str.length();
}

bool StratumClient::process_incoming_messages() {
    if (!connected_ || socket_fd_ < 0) return false;

    char buf[4096];
    while (true) {
        ssize_t n = recv(socket_fd_, buf, sizeof(buf) - 1, 0);
        if (n > 0) {
            buf[n] = '\0';
            recv_buffer_ += buf;
        } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            break; // No more data in buffer
        } else {
            // Connection closed
            connected_ = false;
            return false;
        }
    }

    // Process line by line
    size_t pos;
    while ((pos = recv_buffer_.find('\n')) != std::string::npos) {
        std::string line = recv_buffer_.substr(0, pos);
        recv_buffer_.erase(0, pos + 1);
        if (!line.empty()) {
            handle_line(line);
        }
    }
    return true;
}

void StratumClient::handle_line(const std::string& line) {
    // 1. mining.set_difficulty
    if (line.find("mining.set_difficulty") != std::string::npos) {
        size_t p = line.find("\"params\":");
        if (p != std::string::npos) {
            size_t open_b = line.find('[', p);
            size_t close_b = line.find(']', open_b);
            if (open_b != std::string::npos && close_b != std::string::npos) {
                std::string d_str = line.substr(open_b + 1, close_b - open_b - 1);
                difficulty_ = std::stod(d_str);
            }
        }
    }
    // 2. mining.notify
    else if (line.find("mining.notify") != std::string::npos) {
        StratumJobData job;
        // Simple JSON parameter parsing
        size_t p = line.find("\"params\":");
        if (p != std::string::npos) {
            size_t start = line.find('[', p);
            if (start != std::string::npos) {
                // Parse params array
                std::vector<std::string> tokens;
                bool in_str = false;
                std::string cur;
                int depth = 0;
                for (size_t i = start + 1; i < line.length(); ++i) {
                    char c = line[i];
                    if (c == '\"') in_str = !in_str;
                    else if (c == '[' && !in_str) depth++;
                    else if (c == ']' && !in_str) {
                        if (depth == 0) { if (!cur.empty()) tokens.push_back(cur); break; }
                        depth--;
                    }
                    else if (c == ',' && !in_str && depth == 0) {
                        tokens.push_back(cur);
                        cur.clear();
                        continue;
                    }
                    if (in_str || depth > 0 || (c != ' ' && c != '\"')) cur += c;
                }

                if (tokens.size() >= 8) {
                    job.job_id = tokens[0];
                    job.prevhash_hex = tokens[1];
                    job.coinb1_hex = tokens[2];
                    job.coinb2_hex = tokens[3];
                    job.version_hex = tokens[5];
                    job.nbits_hex = tokens[6];
                    job.ntime_hex = tokens[7];
                    job.clean_jobs = (tokens.size() >= 9 && tokens[8].find("true") != std::string::npos);

                    job.version = (uint32_t)strtoul(job.version_hex.c_str(), nullptr, 16);
                    job.nbits = (uint32_t)strtoul(job.nbits_hex.c_str(), nullptr, 16);
                    job.ntime = (uint32_t)strtoul(job.ntime_hex.c_str(), nullptr, 16);

                    build_header_template(job);

                    if (on_new_job_) on_new_job_(job);
                }
            }
        }
    }
    // 3. Response to share submit / subscribe
    else if (line.find("\"result\":") != std::string::npos) {
        if (line.find("\"result\":true") != std::string::npos) {
            if (on_share_response_) on_share_response_(true, "accepted");
        } else if (line.find("\"result\":false") != std::string::npos) {
            if (on_share_response_) on_share_response_(false, "rejected");
        }
    }
}

void StratumClient::build_header_template(StratumJobData& job) {
    // Construct Coinbase TX (with fixed extranonce2 = 0)
    std::string xn2 = std::string(extranonce2_size_ * 2, '0');
    std::string cb_hex = job.coinb1_hex + extranonce1_ + xn2 + job.coinb2_hex;
    std::vector<uint8_t> cb_bytes = hex_to_bytes(cb_hex);

    uint8_t cb_hash[32];
    sha256_double(cb_bytes.data(), cb_bytes.size(), cb_hash);

    // Calculate Merkle Root
    uint8_t merkle_root[32];
    std::memcpy(merkle_root, cb_hash, 32);

    // Reverse 32-bit word swap of prevhash
    std::vector<uint8_t> prev_bytes = hex_to_bytes(job.prevhash_hex);
    uint8_t prev_swapped[32];
    for (size_t i = 0; i < 32; i += 4) {
        prev_swapped[i + 0] = prev_bytes[i + 3];
        prev_swapped[i + 1] = prev_bytes[i + 2];
        prev_swapped[i + 2] = prev_bytes[i + 1];
        prev_swapped[i + 3] = prev_bytes[i + 0];
    }

    // Assemble 80-byte header
    std::memcpy(job.header_template + 0, &job.version, 4);
    std::memcpy(job.header_template + 4, prev_swapped, 32);
    std::memcpy(job.header_template + 36, merkle_root, 32);
    std::memcpy(job.header_template + 68, &job.ntime, 4);
    std::memcpy(job.header_template + 72, &job.nbits, 4);
    uint32_t zero_nonce = 0;
    std::memcpy(job.header_template + 76, &zero_nonce, 4);
}

bool StratumClient::submit_share(const std::string& job_id, const std::string& extranonce2_hex, uint32_t ntime, uint32_t nonce) {
    message_id_++;
    char ntime_str[16], nonce_str[16];
    snprintf(ntime_str, sizeof(ntime_str), "%08x", ntime);
    snprintf(nonce_str, sizeof(nonce_str), "%08x", nonce);

    std::string submit_req = "{\"id\": " + std::to_string(message_id_) +
                             ", \"method\": \"mining.submit\", \"params\": [\"" +
                             user_ + "\", \"" + job_id + "\", \"" + extranonce2_hex +
                             "\", \"" + ntime_str + "\", \"" + nonce_str + "\"]}\n";
    return send_line(submit_req);
}
