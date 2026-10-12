#pragma once
#include "MotorTypes.hpp"
#include <ecrt.h>

namespace myactua{

struct EthercatBusHealthSnapshot {
    bool master_link_up{false};
    ec_wc_state_t wc_state{EC_WC_ZERO};
    unsigned int working_counter{0};
};

class EthercatAdapter
{
public:
    virtual ~EthercatAdapter() = default;
    virtual bool init(const char* ifname) = 0;  // 初始化网口和ethercat网络

    // 物理层收发：整个网络执行一次
    virtual void receivePhysical() = 0; 
    virtual void sendPhysical() = 0;

    // 逻辑层读写：每个电机对象调用一次（仅操作内存）
    virtual void send(int index, const TxPDO& pdo) = 0;
    virtual RxPDO receive(int index) = 0;
    virtual bool isConfigured(int index) = 0;  // 检查电机是否已经配置完成

    // 默认“不健康”，防止未实现该接口的外部 adapter 被误认为总线正常。
    virtual EthercatBusHealthSnapshot getBusHealth() const { return {}; }
};
}