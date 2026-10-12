#include "EthercatAdapterIGH.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <vector>  
#include <time.h>

#define NSEC_PER_SEC (1000000000L)
#define FREQUENCY 1000
#define PERIOD_NS (NSEC_PER_SEC / FREQUENCY) /*本次设置周期PERIOD_NS为1ms*/

#define CLOCK_TO_USE CLOCK_MONOTONIC  // 使用单调时钟
#define VID_PID          0x00202008,0x00000000   /*Vendor ID, product code	厂商ID和产品代码*/
#define TIMESPEC2NS(T) ((uint64_t) (T).tv_sec * NSEC_PER_SEC + (T).tv_nsec)

namespace myactua {

namespace {

bool parse_diag_enabled_from_env(bool default_value)
{
    const char* env = std::getenv("MYACTUA_ECAT_DIAG");
    if (!env) {
        return default_value;
    }
    if (std::strcmp(env, "0") == 0 ||
        std::strcmp(env, "false") == 0 ||
        std::strcmp(env, "FALSE") == 0 ||
        std::strcmp(env, "off") == 0 ||
        std::strcmp(env, "OFF") == 0) {
        return false;
    }
    return true;
}

uint64_t parse_diag_interval_from_env(uint64_t default_value)
{
    const char* env = std::getenv("MYACTUA_ECAT_DIAG_INTERVAL");
    if (!env) {
        return default_value;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (end == env || parsed == 0) {
        return default_value;
    }
    return static_cast<uint64_t>(parsed);
}

const char* wc_state_to_string(ec_wc_state_t state)
{
    switch (state) {
        case EC_WC_ZERO:
            return "ZERO";
        case EC_WC_INCOMPLETE:
            return "INCOMPLETE";
        case EC_WC_COMPLETE:
            return "COMPLETE";
        default:
            return "UNKNOWN";
    }
}

} // namespace

// ---定义静态配置模板---
ec_pdo_entry_info_t EthercatAdapterIGH::device_pdo_entries[] = {
    {0x6040, 0x00, 16},  // 控制字
    {0x607a, 0x00, 32},  // 目标位置
    {0x60ff, 0x00, 32},  // 目标转速
    {0x6071, 0x00, 16},  // 目标扭矩
    {0x2000, 0x00, 32},  // PVT_KP
    {0x2001, 0x00, 32},  // PVT_KD
    {0x6060, 0x00, 8},   // 设置运行模式
    {0x2ffd, 0x00, 8},   // 填充字节
    /* ========== RxPDO (主站从从站接收) ========== */
    {0x6041, 0x00, 16},   // 状态字
    {0x6064, 0x00, 32},   // 实际位置
    {0x606c, 0x00, 32},   // 实际转速
    {0x6077, 0x00, 16},  // 实际扭矩
    {0x603f, 0x00, 16},  // 错误码
    {0x6061, 0x00, 8},   // 运行模式
    {0x2ffe, 0x00, 8},   // 填充字节
};

ec_pdo_info_t EthercatAdapterIGH::device_pdos[] = {
    // RX 主站发送
    {0x1601, 8, &EthercatAdapterIGH::device_pdo_entries[0]},
    // TX 主站接收
    {0x1a00, 7, &EthercatAdapterIGH::device_pdo_entries[8]},
};

ec_sync_info_t EthercatAdapterIGH::device_syncs[] = {
    {0, EC_DIR_OUTPUT, 0, NULL, EC_WD_DISABLE},
    {1, EC_DIR_INPUT, 0, NULL, EC_WD_DISABLE},
    {2, EC_DIR_OUTPUT, 1, &EthercatAdapterIGH::device_pdos[0], EC_WD_ENABLE},
    {3, EC_DIR_INPUT, 1, &EthercatAdapterIGH::device_pdos[1], EC_WD_DISABLE},
    {0xff, EC_DIR_INVALID, 0, nullptr, EC_WD_DEFAULT}
};    


EthercatAdapterIGH::EthercatAdapterIGH() {
    slave_offsets.resize(kNumSlaves);
    for (std::size_t i = 0; i < kNumSlaves; ++i) {
        slave_configured[i].store(false, std::memory_order_relaxed);
        diag_last_send_cw[i].store(0, std::memory_order_relaxed);
        diag_send_counter[i].store(0, std::memory_order_relaxed);
        tx_shadow[i] = {};
        tx_shadow[i].control_word = CMD_SHUTDOWN;
        tx_shadow[i].target_pos = 0;
        tx_shadow[i].target_vel = 0;
        tx_shadow[i].target_torque = 0;
        tx_shadow[i].pvt_kp = 0;
        tx_shadow[i].pvt_kd = 0;
        tx_shadow[i].op_mode = ControlMode::NONE;
        tx_shadow[i].reserved = 0;
    }
    is_initialized = false;
}    

EthercatAdapterIGH::~EthercatAdapterIGH() {
    if (master) ecrt_release_master(master);
}

bool EthercatAdapterIGH::init(const char* ifname) 
{
    (void)ifname;
    master = ecrt_request_master(0);    // 请求主站控制权
    if (!master) 
    {
        std::cerr << "请求主站失败\n";
        return false;
    }

    domain1 = ecrt_master_create_domain(master); // 创建域
    if (!domain1) 
    {
        std::cerr << "创建域失败\n";
        return false;
    }

    // 配置从站实体
    for (std::size_t i = 0; i < kNumSlaves; i++) 
    {
        const uint16_t position = kSlavePositions[i];
        /* 配置从站索引 */
        sc[i] = ecrt_master_slave_config(master, 0, position, VID_PID);
        if(!sc[i]) 
        {
            std::cerr << "配置从站失败，逻辑索引 " << i
                      << "，物理位置 " << position << "\n";
            return false;
        }
        
        /* 配置从站 PDO */
        if (ecrt_slave_config_pdos(sc[i], EC_END, device_syncs)) 
        {
            std::cerr << "配置从站 PDO 失败，逻辑索引 " << i
                      << "，物理位置 " << position << "\n";
            return false;
        }

        /* 配置从站DC时钟 */
        ecrt_slave_config_dc(sc[i], 0x0300, PERIOD_NS, PERIOD_NS / 2, 0, 0);

        // 注册 PDO 条目到 Domain
        ec_pdo_entry_reg_t reg[] ={
            { 0, position, VID_PID, 0x6040, 0, &slave_offsets[i].off_ctrl_word, nullptr},
            { 0, position, VID_PID, 0x607A, 0, &slave_offsets[i].off_target_pos, nullptr},
            { 0, position, VID_PID, 0x60FF, 0, &slave_offsets[i].off_target_vel, nullptr},
            { 0, position, VID_PID, 0x6071, 0, &slave_offsets[i].off_target_torque, nullptr},
            { 0, position, VID_PID, 0x2000, 0, &slave_offsets[i].off_pvt_kp, nullptr},
            { 0, position, VID_PID, 0x2001, 0, &slave_offsets[i].off_pvt_kd, nullptr},
            { 0, position, VID_PID, 0x6060, 0, &slave_offsets[i].off_mode_of_op, nullptr},

            { 0, position, VID_PID, 0x6041, 0, &slave_offsets[i].off_status_word, nullptr},
            { 0, position, VID_PID, 0x6064, 0, &slave_offsets[i].off_pos, nullptr},
            { 0, position, VID_PID, 0x606C, 0, &slave_offsets[i].off_vel, nullptr},
            {0, position, VID_PID, 0x6077, 0, &slave_offsets[i].off_torque, nullptr},
            {0, position, VID_PID, 0x603F, 0, &slave_offsets[i].off_error, nullptr},
            {0, position, VID_PID, 0x6061, 0, &slave_offsets[i].off_mode_disp, nullptr},
            {0, 0, 0, 0, 0, 0, nullptr, nullptr} // 结束标志
        };
        if (ecrt_domain_reg_pdo_entry_list(domain1, reg)) 
        {
            std::cerr << "注册从站 PDO 条目失败，逻辑索引 " << i
                      << "，物理位置 " << position << "\n";
            return false;
        }
    }

    if(ecrt_master_activate(master))
    {
        std::cerr << "激活主站失败\n";
        return false;
    }

    if(!(domain1_pd = ecrt_domain_data(domain1))) 
    {
        std::cerr << "获取域数据失败\n";
        return false;
    }

    diag_enabled = parse_diag_enabled_from_env(diag_enabled);
    diag_interval_cycles = parse_diag_interval_from_env(diag_interval_cycles);
    std::cout << "[ECAT_DIAG] " << (diag_enabled ? "enabled" : "disabled")
              << ", interval_cycles=" << diag_interval_cycles
              << " (env: MYACTUA_ECAT_DIAG / MYACTUA_ECAT_DIAG_INTERVAL)" << std::endl;

    is_initialized = true;
    return true;    
}

void EthercatAdapterIGH::write_txpdo_to_domain(std::size_t index, const TxPDO& pdo)
{
    if (!domain1_pd || index >= slave_offsets.size()) {
        return;
    }
    SlaveOffsets& off = slave_offsets[index];
    EC_WRITE_U16(domain1_pd + off.off_ctrl_word, pdo.control_word);
    EC_WRITE_S32(domain1_pd + off.off_target_pos, pdo.target_pos);
    EC_WRITE_S32(domain1_pd + off.off_target_vel, pdo.target_vel);
    EC_WRITE_S16(domain1_pd + off.off_target_torque, pdo.target_torque);
    EC_WRITE_S32(domain1_pd + off.off_pvt_kp, pdo.pvt_kp);
    EC_WRITE_S32(domain1_pd + off.off_pvt_kd, pdo.pvt_kd);
    EC_WRITE_S8(domain1_pd + off.off_mode_of_op, pdo.op_mode);
}

void EthercatAdapterIGH::send(int index, const TxPDO& pdo)
{
    if (index < 0 || static_cast<std::size_t>(index) >= slave_offsets.size()) return;

    if (diag_enabled) {
        diag_last_send_cw[index].store(pdo.control_word, std::memory_order_relaxed);
        diag_send_counter[index].fetch_add(1, std::memory_order_relaxed);
    }
    {
        std::lock_guard<std::mutex> lock(tx_shadow_mutex);
        tx_shadow[index] = pdo;
    }
}

RxPDO EthercatAdapterIGH::receive(int index)
{
    RxPDO pdo = {};
    if (index < 0 || static_cast<std::size_t>(index) >= slave_offsets.size()) return pdo;

    SlaveOffsets& off = slave_offsets[index];
    pdo.status_word = EC_READ_U16(domain1_pd + off.off_status_word);
    pdo.pos         = EC_READ_S32(domain1_pd + off.off_pos);
    pdo.vel         = EC_READ_S32(domain1_pd + off.off_vel);
    pdo.torque      = EC_READ_S16(domain1_pd + off.off_torque);
    pdo.error       = EC_READ_U16(domain1_pd + off.off_error);
    pdo.op_mode     = EC_READ_S8 (domain1_pd + off.off_mode_disp);

    return pdo;
}

void EthercatAdapterIGH::receivePhysical() {
    if (!is_initialized || !master || !domain1 || !domain1_pd) {
        return;
    }

    struct timespec time;
    clock_gettime(CLOCK_TO_USE, &time);
    ecrt_master_application_time(master, TIMESPEC2NS(time));

    diag_cycle_counter.fetch_add(1, std::memory_order_relaxed);

    ecrt_master_receive(master);
    ecrt_domain_process(domain1);

    ecrt_master_state(master, &master_state);
    ecrt_domain_state(domain1, &domain1_state);
    health_master_link_up.store(master_state.link_up != 0, std::memory_order_relaxed);
    health_wc_state.store(static_cast<int>(domain1_state.wc_state),
                          std::memory_order_relaxed);
    health_working_counter.store(domain1_state.working_counter,
                                 std::memory_order_relaxed);

    for (std::size_t i = 0; i < kNumSlaves; ++i) {
        ecrt_slave_config_state(sc[i], &sc_state[i]);
        const bool ok = sc_state[i].online && sc_state[i].operational;
        slave_configured[i].store(ok, std::memory_order_relaxed);
    }
}


void EthercatAdapterIGH::sendPhysical() {
    if (!is_initialized || !master || !domain1 || !domain1_pd) {
        return;
    }

    std::array<TxPDO, kNumSlaves> tx_snapshot = {};
    {
        std::lock_guard<std::mutex> lock(tx_shadow_mutex);
        tx_snapshot = tx_shadow;
    }

    for (std::size_t i = 0; i < kNumSlaves; ++i) {
        write_txpdo_to_domain(i, tx_snapshot[i]);
    }

    if (sync_ref_counter) {
        sync_ref_counter--;
    } else {
        sync_ref_counter = 1;
        struct timespec time;
        clock_gettime(CLOCK_TO_USE, &time);
        ecrt_master_sync_reference_clock_to(master, TIMESPEC2NS(time));
    }
    ecrt_master_sync_slave_clocks(master);

    const uint64_t cycle = diag_cycle_counter.load(std::memory_order_relaxed);
    const bool sample_diag = diag_enabled &&
                             (cycle > 0) &&
                             (diag_interval_cycles > 0) &&
                             (cycle % diag_interval_cycles == 0);
    if (sample_diag) {
        ecrt_domain_state(domain1, &domain1_state);
        std::printf("[ECAT_DIAG] cycle=%llu wc=%d wc_state=%s\n",
                    static_cast<unsigned long long>(cycle),
                    domain1_state.working_counter,
                    wc_state_to_string(domain1_state.wc_state));

        for (std::size_t i = 0; i < kNumSlaves; ++i) {
            const SlaveOffsets& off = slave_offsets[i];
            const uint16_t app_cw = diag_last_send_cw[i].load(std::memory_order_relaxed);
            const uint32_t app_send_cnt = diag_send_counter[i].load(std::memory_order_relaxed);
            const uint16_t pd_cw = EC_READ_U16(domain1_pd + off.off_ctrl_word);
            const uint16_t sw = EC_READ_U16(domain1_pd + off.off_status_word);
            const uint16_t err = EC_READ_U16(domain1_pd + off.off_error);
            const int8_t op = EC_READ_S8(domain1_pd + off.off_mode_disp);
            const int8_t tgt_mode = EC_READ_S8(domain1_pd + off.off_mode_of_op);
            std::printf(
                "  M%zu send_cw=0x%04X send_cnt=%u pd_cw=0x%04X"
                " status=0x%04X err=0x%04X tgt_mode=%d op=%d cfg=%d\n",
                i,
                static_cast<unsigned>(app_cw),
                static_cast<unsigned>(app_send_cnt),
                static_cast<unsigned>(pd_cw),
                static_cast<unsigned>(sw),
                static_cast<unsigned>(err),
                static_cast<int>(tgt_mode),
                static_cast<int>(op),
                slave_configured[i].load(std::memory_order_relaxed) ? 1 : 0);
        }
        std::fflush(stdout);
    }

    ecrt_domain_queue(domain1);
    ecrt_master_send(master);
}

// 检查特定从站的配置状态
bool EthercatAdapterIGH::isConfigured(int index) {
    if (index < 0 || index >= static_cast<int>(kNumSlaves)) {
        return false;
    }
    return slave_configured[index].load(std::memory_order_relaxed);
}

EthercatBusHealthSnapshot EthercatAdapterIGH::getBusHealth() const
{
    EthercatBusHealthSnapshot health;
    health.master_link_up = health_master_link_up.load(std::memory_order_relaxed);
    health.wc_state = static_cast<ec_wc_state_t>(
        health_wc_state.load(std::memory_order_relaxed));
    health.working_counter = health_working_counter.load(std::memory_order_relaxed);
    return health;
}


} // namespace myactua
