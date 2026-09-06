/**
 * @file stratum_client.cpp
 * @brief Lightweight Stratum v1 TCP client implementation for Blake2bCudaMiner.
 */

#include "stratum_client.h"
#include "blake2b_host.h"
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
      extranonce2_size_(4), message_id_(10) {}

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
    std::string sub_req = "{\"id\": 1, \"method\": \"mining.subscribe\", \"params\": [\"Blake2bCudaMiner/1.3.1\"]}\n";
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
    // Check if this is a response to a client request (id: 1, 2, 3...)
    size_t id_pos = line.find("\"id\":");
    if (id_pos != std::string::npos && line.find("\"id\":null") == std::string::npos && line.find("\"id\": null") == std::string::npos) {
        int id_val = std::atoi(line.c_str() + id_pos + 5);
        if (id_val == 1) {
            // Subscribe response: extract extranonce1 (or sid) and extranonce2_size
            // e.g. [[["mining.notify", "sid1"], ...], "00000000645a6424", 8]
            size_t res_pos = line.find("\"result\":");
            if (res_pos != std::string::npos) {
                size_t start_arr = line.find('[', res_pos);
                if (start_arr != std::string::npos) {
                    int depth = 0;
                    size_t end_arr = std::string::npos;
                    for (size_t i = start_arr; i < line.length(); ++i) {
                        if (line[i] == '[') depth++;
                        else if (line[i] == ']') {
                            depth--;
                            if (depth == 0) { end_arr = i; break; }
                        }
                    }
                    if (end_arr != std::string::npos) {
                        std::string res_arr = line.substr(start_arr, end_arr - start_arr + 1);
                        size_t c = res_arr.rfind(',');
                        if (c != std::string::npos) {
                            int sz = std::atoi(res_arr.substr(c + 1, res_arr.length() - c - 2).c_str());
                            if (sz > 0 && sz <= 16) extranonce2_size_ = sz;
                            size_t q2 = res_arr.rfind('\"', c);
                            if (q2 != std::string::npos && q2 > 0) {
                                size_t q1 = res_arr.rfind('\"', q2 - 1);
                                if (q1 != std::string::npos) {
                                    extranonce1_ = res_arr.substr(q1 + 1, q2 - q1 - 1);
                                }
                            }
                        }
                    }
                }
            }
            std::cout << "  • Subscribed: extranonce1=" << extranonce1_ << ", extranonce2_size=" << extranonce2_size_ << std::endl;
        } else if (id_val > 2) {
            // Actual share submission response (id > 2)
            if (line.find("\"result\":true") != std::string::npos) {
                std::cout << "  ✅ Share ACCEPTED: " << line << std::endl;
                if (on_share_response_) on_share_response_(true, "accepted");
            } else {
                std::cerr << "  ❌ Share REJECTED: " << line << std::endl;
                if (on_share_response_) on_share_response_(false, "rejected");
            }
        }
        return;
    }

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
                    if (c == '\"') {
                        in_str = !in_str;
                        continue;
                    }
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
                    if (in_str || depth > 0 || c != ' ') cur += c;
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
}

void StratumClient::build_header_template(StratumJobData& job) {
    if (job.prevhash_hex.length() == 160) {
        // Bitcoin Knots Profile 0 Solo: Full 80-byte precomputed ASIC message directly provided
        std::vector<uint8_t> h = hex_to_bytes(job.prevhash_hex);
        if (h.size() == 80) {
            std::memcpy(job.header_template, h.data(), 80);
            job.is_ratum = false;
            return;
        }
    }

    if (job.prevhash_hex.length() == 64) {
        // RATUM / SiaMining Dialect (Bitcoin Knots v2 via ratum-gateway)
        job.is_ratum = true;

        // 1. Prevhash (32 bytes)
        std::vector<uint8_t> prev_bytes = hex_to_bytes(job.prevhash_hex);
        if (prev_bytes.size() != 32) return;

        // 2. Extranonce (16 bytes): decode extranonce1 (8 bytes) + pad with 8 zero bytes for extranonce2
        std::vector<uint8_t> extranonce = hex_to_bytes(extranonce1_);
        extranonce.resize(16, 0);

        // 3. Leaf buffer: 0x00 + coinb1 + extranonce + coinb2
        std::vector<uint8_t> coinb1_bytes = hex_to_bytes(job.coinb1_hex);
        std::vector<uint8_t> coinb2_bytes = hex_to_bytes(job.coinb2_hex);

        std::vector<uint8_t> leaf_buf;
        leaf_buf.reserve(1 + coinb1_bytes.size() + extranonce.size() + coinb2_bytes.size());
        leaf_buf.push_back(0x00);
        leaf_buf.insert(leaf_buf.end(), coinb1_bytes.begin(), coinb1_bytes.end());
        leaf_buf.insert(leaf_buf.end(), extranonce.begin(), extranonce.end());
        leaf_buf.insert(leaf_buf.end(), coinb2_bytes.begin(), coinb2_bytes.end());

        // 4. Calculate hash1 = blake2b_256(leaf_buf)
        uint8_t hash1_32[32];
        blake2b_256(leaf_buf.data(), leaf_buf.size(), hash1_32);

        // 5. Assemble 80-byte header template
        std::memset(job.header_template, 0, 80);
        // Bytes 0..31: prevhash
        std::memcpy(job.header_template + 0, prev_bytes.data(), 32);
        // Bytes 32..35: nonce (0)
        // Bytes 36..39: nonce2 (0)
        // Bytes 40..47: ntime (8 bytes from job.ntime_hex)
        std::vector<uint8_t> ntime_bytes = hex_to_bytes(job.ntime_hex);
        if (ntime_bytes.size() == 8) {
            std::memcpy(job.header_template + 40, ntime_bytes.data(), 8);
        } else {
            std::memcpy(job.header_template + 40, &job.ntime, 4);
        }
        // Bytes 48..79: hash1 (32 bytes)
        std::memcpy(job.header_template + 48, hash1_32, 32);
        return;
    }

    // Fallback: Construct Coinbase TX (with fixed extranonce2 = 0)
    job.is_ratum = false;
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
    uint8_t prev_swapped[32] = {0};
    for (size_t i = 0; i < 32 && i + 3 < prev_bytes.size(); i += 4) {
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
    return submit_share(job_id, extranonce2_hex, "", ntime, (uint64_t)nonce, false);
}

bool StratumClient::submit_share(const std::string& job_id, const std::string& extranonce2_hex, const std::string& ntime_hex, uint32_t ntime, uint64_t found_nonce64, bool is_ratum) {
    message_id_++;
    std::string submit_req;

    if (is_ratum) {
        // RATUM Gateway submission format:
        // params: [user, job_id, extranonce2 (16 hex), ntime_hex (16 hex), nonce (16 hex: nonce LE + nonce2 LE)]
        std::string en2_hex = extranonce2_hex;
        if (en2_hex.length() < 16) {
            en2_hex = std::string(16 - en2_hex.length(), '0') + en2_hex;
        }

        uint8_t nonce_bytes[8];
        std::memcpy(nonce_bytes, &found_nonce64, 8);
        char nonce_hex[17];
        for (int i = 0; i < 8; ++i) {
            snprintf(nonce_hex + i * 2, 3, "%02x", nonce_bytes[i]);
        }

        std::string nt_hex = ntime_hex;
        if (nt_hex.empty()) {
            char buf[17];
            snprintf(buf, sizeof(buf), "%08x00000000", ntime);
            nt_hex = buf;
        }

        submit_req = "{\"id\": " + std::to_string(message_id_) +
                     ", \"method\": \"mining.submit\", \"params\": [\"" +
                     user_ + "\", \"" + job_id + "\", \"" + en2_hex +
                     "\", \"" + nt_hex + "\", \"" + nonce_hex + "\"]}\n";
    } else {
        // Legacy / Solo submission format
        char ntime_str[16], nonce_str[16];
        snprintf(ntime_str, sizeof(ntime_str), "%08x", ntime);
        snprintf(nonce_str, sizeof(nonce_str), "%08x", (uint32_t)found_nonce64);

        submit_req = "{\"id\": " + std::to_string(message_id_) +
                     ", \"method\": \"mining.submit\", \"params\": [\"" +
                     user_ + "\", \"" + job_id + "\", \"" + extranonce2_hex +
                     "\", \"" + ntime_str + "\", \"" + nonce_str + "\"]}\n";
    }

    std::cout << "  ➡️ [SUBMIT] " << submit_req;
    return send_line(submit_req);
}
