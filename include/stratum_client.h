#ifndef STRATUM_CLIENT_H
#define STRATUM_CLIENT_H

/**
 * @file stratum_client.h
 * @brief Lightweight TCP Stratum v1 client for Bitcoin Knots (Blake2bCudaMiner).
 */

#include <string>
#include <vector>
#include <functional>
#include <cstdint>

struct StratumJobData {
    std::string job_id;
    std::string prevhash_hex;
    std::string coinb1_hex;
    std::string coinb2_hex;
    std::vector<std::string> merkle_branches;
    std::string version_hex;
    std::string nbits_hex;
    std::string ntime_hex;
    bool clean_jobs;
    uint32_t nbits;
    uint32_t ntime;
    uint32_t version;
    uint8_t header_template[80]; // 80-byte base header template
};

class StratumClient {
public:
    using JobCallback = std::function<void(const StratumJobData&)>;
    using ShareResponseCallback = std::function<void(bool accepted, const std::string& message)>;

    StratumClient(const std::string& host, int port, const std::string& user, const std::string& pass);
    ~StratumClient();

    bool connect_to_server();
    void disconnect_server();
    bool is_connected() const { return connected_; }

    void set_job_callback(JobCallback cb) { on_new_job_ = cb; }
    void set_response_callback(ShareResponseCallback cb) { on_share_response_ = cb; }

    bool process_incoming_messages();
    bool submit_share(const std::string& job_id, const std::string& extranonce2_hex, uint32_t ntime, uint32_t nonce);

    double get_difficulty() const { return difficulty_; }

private:
    std::string host_;
    int port_;
    std::string user_;
    std::string pass_;
    int socket_fd_;
    bool connected_;
    double difficulty_;
    std::string extranonce1_;
    int extranonce2_size_;
    int message_id_;
    std::string recv_buffer_;

    JobCallback on_new_job_;
    ShareResponseCallback on_share_response_;

    bool send_line(const std::string& json_str);
    void handle_line(const std::string& line);
    void build_header_template(StratumJobData& job);
};

#endif // STRATUM_CLIENT_H
