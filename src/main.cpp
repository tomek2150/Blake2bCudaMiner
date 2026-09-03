/**
 * @file main.cpp
 * @brief Standalone Blake2b GPU Miner CLI for Bitcoin Knots (Blake2bCudaMiner).
 */

#include "blake2b_cuda.cuh"
#include "blake2b_host.h"
#include "stratum_client.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <atomic>
#include <csignal>
#include <unistd.h>

extern "C" cudaError_t blake2b_set_midstate_cuda(const blake2b_midstate_t* host_midstate);
extern "C" cudaError_t blake2b_launch_kernel(
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint64_t target_diff,
    uint32_t* d_found_nonces,
    uint32_t* d_found_count,
    uint32_t block_size,
    cudaStream_t stream
);

static std::atomic<bool> g_running(true);

void signal_handler(int) {
    g_running = false;
}

/**
 * @brief Displays the command-line help screen.
 */
void print_help(const char* prog_name) {
    std::cout << R"ASCII(
  .............................:... ............   .--:-:.......................
.........................:-=:   .:....:.::-:.::.   :-...   .::..................
......................::.  .--.  .:..:::------:::::.=+=:.     .:................
...................::.  . .-=-.::--=------------=----:::-+-. .   .:.............
...................   ..=--::---=-:.............=-:::::--::==: . ...............
............ ...:    .--::=-::..-:              .=.--:.::::::+:... .............
............. ..    .=..::.:.--.-::            .-::::......:-:*.....::..........
...............    :+::... .. -:--:.          . ---:-.   ...-:+-:. .-=-:........
...............    :=:... .  .=.-::.  ..........:--:=.   ...::=-....::::-:......
.............:.    -=.....   .=:=:=:...::::----=----+.  .. .::=:....-=-.=+=:::..
............--:    .+::..    .=-=..-=--:--=====-=+---.. .. ..==....:.:::-:=+::::
..........:=:.::    --...     .-=.:=-==--====+==-++-::    ..:-.. .-= ..:-==+*-::
........:-=-- .#:. .:=:.:   ...:..=:+=+---==-====*+--:.....:.... -=-:=: -:--+*-:
........:+-.-+-.--.....:.. ..::..:+---+---==-++==+-:... .:-.....==-.:--+-::-**+:
.....::-=-:+=:. :=-.::......::....=--.+-::--:=+=:=-...:--:...::-.-.....:--+=++#*
...::::##+=.-.  .+--+=-::::-::....-=:=--..:..-*:.:-.=-+=::..:=-:--:--..:+-===**=
...:::=**+-:. :-.==-::=--:::::---.:-..==. ...:#:.:==+=---::-=--=-:..:==.:=+**::-
...:.-+---:..=-:.-*+-=-==-:---:---+=-::: .....:-==---+=---:-+:::-::.:-=+-++***-=
..::.--*=-:.=:::::+--+--++==-==--:::-===-====--:::::::==::-==+-=-::::.:-+=-+=++-
....:=++++-*::..:----++--=+---=--::::::------=:::::-=-::-=======-::-::-++%-:--==
...:-=+=:=+:-:..-:::::-=:=+=--::-==-::..::::...:::==:--==*##=---=-:-=-:=++*-:=-+
...:*--::-*++=:=:.::::=+===++=-==--==-::......::-=---=*#%%%*----+-:::=-::==+-:=-
..:-+::::====:--:...:.*=---+#%%#*+==-==-::::::::=-=*#%%%%%*+==:-*:::.:=::.-*+:-:
...=.-::-=-:::::...::.+*:--=*#%%%%%#*===::::::::=*%%%%%%#*++--:+-:....:-.:-=+-:-
..--=:::+---:=--..:...-#=:-=+=+##%%%%%#*:::::::-+*####+=+++--:==-:.::..-::::+=--
:.---:-:--=.-::..:...:.-*=::-=====++++==----------:--==+=-:::-:::.:.:..:.::---:-
::=+.---=::.-::....::-:::-=----=-=-=---=++====---==--::::::::--.-.:.:.:.:::---:-
:--:--:---::=*+:::..:-..::-::.:::::--*+=.::......:=--:..::::-==::.....::::::-+:-
:==-:-:=:-:::....-:.:-:.::=+-.:.....:.+-..........-=-::.:::=::-+:. ...--::--=**+
:+=-:-:==-:--...:*=.:+-.:-+=----::...:+-...........+-::::-=:-:::-  ..:==::--++#=
.+++=::=-:.:-:-..=+:.-=:.-=-::---::...=:.....   ...+::::-=:-.--::. ..+=:.:::-:--
:+-:=:-=-::-:.-.:+=:.:--.:==:::-:--:.:-- . .       =-:-::--..::.   .:::.---:::=+
-==--:-=-::-::-..:=*:.==-.:-=-:-==+--:-- .         --:-:...::-:    .:-:-+*=-:-=*
:---=:-+-.-:-::-..-=-.=+ .=.-=+-=.-==-==           :=--:...-..    :=--:=*+:.-:=:
:---+:-=..---.-::-.:-.:: -.. .-++::-==-+           .=.-:....     :**+.:---:::-=:
-=:=*+#+..::-::-.   .:. =:    .:-=:=++-+.          .-.::...      -++:.=:-=::-==-
:=--+-:=:::.=*+#:   .: .=.     ....::+==:           -..    .   .:-:.:-.:=::.---.
-=--+:-=*+- :++#-   .: .=.      ...    :-.  .. ......          -+-::-.-+=-.:=-:.
-=--+::+##* :=-=:   .:---       ....                          .-=::=--++*-.++=:.
+=--*-:-=++ .=:-:   .                                        .--.:+*---==.:++=..
=---=-::-:-:.--::   . .:.=+.                              =-.:=-.-=-.==-..---:. 
-==+=-----:: .-::. -=-::.=:.                             --.:==.-=+:-++: .-=:   
-+++-=-:--:=  -::..-.-::.=:.                            =.-.-=.:=-::=+-  .==:   
:=*+.==-++=-. -=-..::.:. =:.   .:..           ..:- .  :*-=.:=:.-+-.:==.  :=:    
-=+=.=++==--. -*=:..::.:..-=-.....:..      ....  ..---:=-..--:.=+..==::..--.    
=-=-:-+=::==: .-::.. . ..:----==:--.      .:====-=--:.    .:..-=:.:=:--..-:     
.-==.:--::-+: :==:   .::.....:.+:.+.     .=+ +-.......-=:.....+-.:-=----:+:     
 :==..--:.-+:.:-:::::::::::::::-::. .: .:.::.::..::.::-.   ..==.:--:::..:*.  .:+
 :=-. ---.-==-=:-+-::::. ..::.......-..:....=--......-::.=:--*-..       :+---:-+
.:==. :==.:++--.-+:.=-..::.:.:-:....= .:...:=-:-.  . .::-.-.=.        .=+-----. 
-:-=  .==:.==*-::=...:.:::::......... .:.::::. .-*.     --::-.    ..:--:--..... 
::-=   -+:.+*#=.:-....::: .  .=:=:.    .-=.--:....*-.  .:-..-   .-*+-==::   ..-:
===-   :+:.+#%+::=::.::.. ..:--:+:.. ...--:+-.-: ..-=  ..:..-   .:::-.      :.:.


      )ASCII" << "\n";
    std::cout << "================================================================================\n";
    std::cout << " ⚡ Blake2bCudaMiner: High-Efficiency Blake2b GPU Miner v.1\n";
    std::cout << "================================================================================\n";
    std::cout << "Usage: " << prog_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -o, --url <url>          Stratum server URL (default: 127.0.0.1:3333)\n";
    std::cout << "  -u, --user <username>    Stratum username or payout address (default: miner)\n";
    std::cout << "  -p, --pass <password>    Stratum password (default: x)\n";
    std::cout << "  -d, --device <id>        CUDA device ID (default: 0)\n";
    std::cout << "  -b, --block-size <n>     Threads per CUDA block [64, 128, 256, 512] (default: 512)\n";
    std::cout << "  -h, --help               Display this help message and exit\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog_name << " -o stratum+tcp://127.0.0.1:3333 -u miner -p x\n";
    std::cout << "  " << prog_name << " -o stratum+tcp://pool.example.com:3333 -u <wallet_address> -p x -d 0 -b 512\n";
    std::cout << "============================================================\n";
}

int main(int argc, char** argv) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::string pool_url = "127.0.0.1:3333";
    std::string user = "miner";
    std::string pass = "x";
    int device_id = 0;
    uint32_t block_size = 512;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help" || arg == "-help") {
            print_help(argv[0]);
            return 0;
        } else if ((arg == "-o" || arg == "--url") && i + 1 < argc) {
            pool_url = argv[++i];
        } else if ((arg == "-u" || arg == "--user") && i + 1 < argc) {
            user = argv[++i];
        } else if ((arg == "-p" || arg == "--pass") && i + 1 < argc) {
            pass = argv[++i];
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            device_id = std::stoi(argv[++i]);
        } else if ((arg == "-b" || arg == "--block-size") && i + 1 < argc) {
            block_size = (uint32_t)std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << "\n";
            std::cerr << "Use -h or --help for available options.\n";
            return 1;
        }
    }

    // Parse URL (Host and Port)
    if (pool_url.find("stratum+tcp://") == 0) pool_url = pool_url.substr(14);
    std::string host = "127.0.0.1";
    int port = 3333;
    size_t colon = pool_url.find(':');
    if (colon != std::string::npos) {
        host = pool_url.substr(0, colon);
        port = std::stoi(pool_url.substr(colon + 1));
    }

    std::cout << "============================================================" << std::endl;
    std::cout << " ⚡ Blake2bCudaMiner: High-Efficiency Blake2b GPU Miner v1.0" << std::endl;
    std::cout << "============================================================" << std::endl;

    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::cout << "  • GPU #" << device_id << ":              " << prop.name << std::endl;
    std::cout << "  • Streaming Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "  • Connecting to Stratum:     " << host << ":" << port << std::endl;
    std::cout << "  • Username:                  " << user << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    // Allocate GPU buffers
    uint32_t* d_found_nonces;
    uint32_t* d_found_count;
    cudaMalloc(&d_found_nonces, 16 * sizeof(uint32_t));
    cudaMalloc(&d_found_count, sizeof(uint32_t));
    cudaMemset(d_found_count, 0, sizeof(uint32_t));

    StratumClient stratum(host, port, user, pass);
    std::atomic<bool> new_job_ready(false);
    StratumJobData current_job;
    blake2b_midstate_t current_midstate;

    std::atomic<uint64_t> total_hashes(0);
    std::atomic<uint32_t> accepted_shares(0);
    std::atomic<uint32_t> rejected_shares(0);

    stratum.set_job_callback([&](const StratumJobData& job) {
        current_job = job;
        blake2b_precompute_midstate(job.header_template, job.nbits, &current_midstate);
        blake2b_set_midstate_cuda(&current_midstate);
        new_job_ready = true;
    });

    stratum.set_response_callback([&](bool accepted, const std::string&) {
        if (accepted) accepted_shares++;
        else rejected_shares++;
    });

    if (!stratum.connect_to_server()) {
        std::cerr << "❌ Error: Connection to " << host << ":" << port << " failed!" << std::endl;
        cudaFree(d_found_nonces);
        cudaFree(d_found_count);
        return 1;
    }

    std::cout << "  ✅ Stratum connected! Waiting for first block template..." << std::endl;

    const uint32_t batch_size = 64 * 1024 * 1024; // 67,108,864 nonces per launch
    uint32_t nonce_counter = 0;
    auto last_stats_time = std::chrono::steady_clock::now();
    uint64_t last_hash_count = 0;

    while (g_running) {
        stratum.process_incoming_messages();

        if (!new_job_ready) {
            usleep(5000);
            continue;
        }

        // Calculate target difficulty threshold from Stratum difficulty
        double diff = stratum.get_difficulty();
        uint64_t target_diff = (uint64_t)(0x00000000FFFF0000ULL / diff);
        if (target_diff == 0) target_diff = 0x00000000FFFF0000ULL;

        // Launch mining kernel
        cudaMemset(d_found_count, 0, sizeof(uint32_t));
        blake2b_launch_kernel(
            nonce_counter,
            batch_size,
            target_diff,
            d_found_nonces,
            d_found_count,
            block_size,
            0
        );
        cudaDeviceSynchronize();

        total_hashes += batch_size;
        nonce_counter += batch_size;

        // Check for found nonces
        uint32_t found_cnt = 0;
        cudaMemcpy(&found_cnt, d_found_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (found_cnt > 0) {
            if (found_cnt > 16) found_cnt = 16;
            std::vector<uint32_t> found_nonces(found_cnt);
            cudaMemcpy(found_nonces.data(), d_found_nonces, found_cnt * sizeof(uint32_t), cudaMemcpyDeviceToHost);

            for (uint32_t fn : found_nonces) {
                stratum.submit_share(current_job.job_id, "00000000", current_job.ntime, fn);
            }
        }

        // Live statistics every 5 seconds
        auto now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - last_stats_time).count();
        if (elapsed_sec >= 5.0) {
            double cur_mhs = ((total_hashes - last_hash_count) / elapsed_sec) / 1e6;
            std::cout << "[INFO] GPU #0: " << std::fixed << std::setprecision(2)
                      << cur_mhs << " MH/s (" << cur_mhs / 1000.0 << " GH/s)"
                      << " | Shares: " << accepted_shares.load() << "/" << (accepted_shares + rejected_shares)
                      << " (Diff " << stratum.get_difficulty() << ")" << std::endl;
            last_stats_time = now;
            last_hash_count = total_hashes.load();
        }
    }

    std::cout << "\nShutting down Blake2bCudaMiner..." << std::endl;
    stratum.disconnect_server();
    cudaFree(d_found_nonces);
    cudaFree(d_found_count);

    return 0;
}
