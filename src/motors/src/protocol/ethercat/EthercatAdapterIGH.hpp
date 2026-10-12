#pragma once
#include "EthercatAdapter.hpp"
#include <ecrt.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <atomic>
#include <mutex>

namespace myactua {

/*
 * 0: 有转发站版本（默认）
 * 1: 无转发站版本（仅电机从站）
 */
#ifndef MYACTUA_ECAT_NO_FORWARDERS
#define MYACTUA_ECAT_NO_FORWARDERS 0
#endif

/*
 * 从站物理位置映射（逻辑索引 -> EtherCAT position）
 *
 * 默认: 使用当前现场拓扑（包含转发站时的历史映射）
 * 当 MYACTUA_ECAT_NO_FORWARDERS == 1:
 * 使用“无转发站，仅电机从站”的连续映射
 */
#if MYACTUA_ECAT_NO_FORWARDERS
#define MYACTUA_NO_FORWARDERS_MOTORS_NUM 1
inline constexpr std::array<uint16_t, MYACTUA_NO_FORWARDERS_MOTORS_NUM> kSlavePositions = {
    0
};

#else
inline constexpr std::array<uint16_t, 12> kSlavePositions = {
    1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13
};
#define MYACTUA_FORWARDERS_MOTORS_NUM 12
#endif

inline constexpr std::size_t kNumSlaves = kSlavePositions.size();


/* 继承 class EthercatAdapter */
class EthercatAdapterIGH : public EthercatAdapter {
private:
    struct SlaveOffsets
    {
        unsigned int off_ctrl_word;     // 0x6040:00
        unsigned int off_target_pos;    // 0x607A:00
        unsigned int off_target_vel;    // 0x60FF:00
        unsigned int off_target_torque; // 0x6071:00
        unsigned int off_pvt_kp;        // 0x2000:00
        unsigned int off_pvt_kd;        // 0x2001:00
        unsigned int off_mode_of_op;    // 0x6060:00
        unsigned int off_reserved_tx;   // 0x2FFD:00
        unsigned int off_status_word;   // 0x6041:00
        unsigned int off_pos;           // 0x6064:00
        unsigned int off_vel;           // 0x606C:00
        unsigned int off_torque;        // 0x6077:00
        unsigned int off_error;         // 0x603F:00
        unsigned int off_mode_disp;     // 0x6061:00
        unsigned int off_reserved_rx;   // 0x2FFE:00
    };

    // 定义映射模板
    static ec_pdo_entry_info_t  device_pdo_entries[];
    static ec_pdo_info_t        device_pdos[];
    static ec_sync_info_t       device_syncs[];

    ec_master_t *master = NULL;
    ec_master_state_t master_state = {};

    ec_domain_t *domain1 = NULL;
    ec_domain_state_t domain1_state = {};
    uint8_t *domain1_pd = nullptr; 

    std::array<ec_slave_config_t *, kNumSlaves> sc = {};
    std::array<ec_slave_config_state_t, kNumSlaves> sc_state = {};
    std::array<std::atomic<bool>, kNumSlaves> slave_configured = {};
    std::atomic<bool> health_master_link_up{false};
    std::atomic<int> health_wc_state{static_cast<int>(EC_WC_ZERO)};
    std::atomic<unsigned int> health_working_counter{0};

    unsigned int sync_ref_counter = 0;
    bool is_initialized = false;
    
    std::vector<SlaveOffsets> slave_offsets;


    // 应用层先写 shadow，sendPhysical() 在控制主循环内统一落盘到 domain1_pd
    std::array<TxPDO, kNumSlaves> tx_shadow = {};
    std::mutex tx_shadow_mutex;
    void write_txpdo_to_domain(std::size_t index, const TxPDO& pdo);

    
    // 诊断: 仅用于验证时序问题，不改变控制路径
    bool diag_enabled = false;
    uint64_t diag_interval_cycles = 500;
    std::atomic<uint64_t> diag_cycle_counter{0};
    std::array<std::atomic<uint16_t>, kNumSlaves> diag_last_send_cw = {};
    std::array<std::atomic<uint32_t>, kNumSlaves> diag_send_counter = {};

public:
    EthercatAdapterIGH();
    ~EthercatAdapterIGH() override;

    bool  init(const char* ifname) override;
    void  send(int index, const TxPDO& pdo) override;
    RxPDO receive(int index) override;

    void receivePhysical() override;
    void sendPhysical() override;
    bool isConfigured(int index) override;
    EthercatBusHealthSnapshot getBusHealth() const override;
};

}
