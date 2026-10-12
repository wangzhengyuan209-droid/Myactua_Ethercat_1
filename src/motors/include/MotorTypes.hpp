#pragma once
#include <cstdint>

namespace myactua {

constexpr uint32_t TX_CONTROL_WORD_OFFSET = 0;  // 2 bytes
constexpr uint32_t TX_TARGET_POS_OFFSET   = 2;  // 4 bytes
constexpr uint32_t TX_TARGET_VEL_OFFSET   = 6;  // 4 bytes
constexpr uint32_t TX_TARGET_TORQUE_OFFSET= 10; // 2 bytes
constexpr uint32_t TX_PVT_KP_OFFSET       = 12; // 4 bytes
constexpr uint32_t TX_PVT_KD_OFFSET       = 16; // 4 bytes
constexpr uint32_t TX_MODE_OF_OP_OFFSET   = 20; // 1 bytes
constexpr uint32_t TX_RESERVED_OFFSET     = 21; // 1 bytes
constexpr uint32_t TX_PDO_SIZE = 22;

constexpr uint32_t RX_STATUS_WORD_OFFSET  = 0;   // 2 bytes
constexpr uint32_t RX_POS_OFFSET          = 2;   // 4 bytes
constexpr uint32_t RX_VEL_OFFSET          = 6;   // 4 bytes
constexpr uint32_t RX_TORQUE_OFFSET       = 10;  // 2 bytes
constexpr uint32_t RX_ERROR_OFFSET        = 12;  // 2 bytes
constexpr uint32_t RX_MODE_DISP_OFFSET    = 14;  // 1 bytes
constexpr uint32_t RX_RESERVED_OFFSET     = 15;  // 1 bytes
constexpr uint32_t RX_PDO_SIZE = 16;


/* 运行模式:0x6060 */
enum ControlMode : int8_t{
    NONE = 0,
    PVT  = 0x05,  // PVT/MIT 混合控制模式
    MIT  = PVT,   // MIT 模式别名
    CSP  = 0x08,  // 周期同步位置模式
    CSV  = 0x09,  // 周期同步速度模式
    CST  = 0x0A,  // 周期同步扭矩模式
};


/* 定义状态字位 */
enum StatusWordBits : uint16_t {
    BIT_READY_TO_SWITCH_ON = 0x0001,
    BIT_SWITCHED_ON = 0x0002,        // 伺服使能
    BIT_OPERATION_ENABLED = 0x0004,  // 伺服运行
    BIT_FAULT = 0x0008,              // 故障
    BIT_VOLTAGE_ENABLED = 0x0010,    // 接通电路
    BIT_QUICK_STOP = 0x0020,         // 快速停止
    BIT_NOT_FAULT = 0x0040,          // 伺服无故障
    BIT_WARNING = 0x0080,            // 警告
    BIT_MANUFACTURER_8 = 0x0100,
    BIT_REMOTE = 0x0200,
    BIT_TARGET_REACHED = 0x0400,
    BIT_INTERNAL_LIMIT_ACTIVE = 0x0800,
    BIT_OPERATION_MODE_SPECIFIC_0 = 0x1000,
    BIT_OPERATION_MODE_SPECIFIC_1 = 0x2000,
    BIT_MANUFACTURER_14 = 0x4000,
    BIT_MANUFACTURER_15 = 0x8000,
};

inline bool is_ready_to_switch_on(uint16_t sw){return (sw & BIT_READY_TO_SWITCH_ON) != 0;}
inline bool is_switched_on(uint16_t sw){return (sw & BIT_SWITCHED_ON) != 0;} 
inline bool is_operation_enabled(uint16_t sw){return (sw & BIT_OPERATION_ENABLED) != 0;} 
inline bool is_fault(uint16_t sw){return (sw & BIT_FAULT) != 0;}               
inline bool is_voltage_enabled(uint16_t sw){return (sw & BIT_VOLTAGE_ENABLED) != 0;}
inline bool is_quick_stop_bit(uint16_t sw){return (sw & BIT_QUICK_STOP) != 0;}
inline bool is_not_fault(uint16_t sw){return (sw & BIT_NOT_FAULT) != 0;}
inline bool is_warning_bit(uint16_t sw){return (sw & BIT_WARNING) != 0;}
inline bool is_remote_bit(uint16_t sw){return (sw & BIT_REMOTE) != 0;}
inline bool is_target_reached_bit(uint16_t sw){return (sw & BIT_TARGET_REACHED) != 0;}
inline bool is_internal_limit_active_bit(uint16_t sw){return (sw & BIT_INTERNAL_LIMIT_ACTIVE) != 0;}


/* 定义控制字位 (0x6040) */
enum ControlWordBits : uint16_t {
    CW_BIT_SERVO_ENABLE = 0x0001,         // Bit 0: 伺服使能 (1-有效, 0-无效)
    CW_BIT_VOLTAGE_ENABLE = 0x0002,       // Bit 1: 接通主回路电路 (1-有效, 0-无效)
    CW_BIT_QUICK_STOP = 0x0004,           // Bit 2: 快速停机 (0-有效快速停机, 1-无效)
    CW_BIT_OPERATION_ENABLE = 0x0008,     // Bit 3: 伺服运行 (1-有效, 0-无效)
    CW_BIT_MODE_SPECIFIC_0 = 0x0010,      // Bit 4: 特定运行模式
    CW_BIT_MODE_SPECIFIC_1 = 0x0020,      // Bit 5: 特定运行模式
    CW_BIT_MODE_SPECIFIC_2 = 0x0040,      // Bit 6: 特定运行模式
};

/* 常用控制字命令组合 */
enum ControlWordCommand : uint16_t {
    CMD_SHUTDOWN = 0x0006,           // 0110：失能，不触发快速停机
    CMD_QUICK_STOP = 0x0002,         // 0010：失能并急停
    CMD_DISABLE_VOLTAGE = 0x0000,    // 0000：停电

    CMD_SWITCH_ON = 0x0007,          // 0111：保持使能但停止运行
    CMD_DISABLE_OPERATION = 0x0007,  // 0111：保持使能，停止运行
    CMD_ENABLE_OPERATION = 0x000F,   // 1111：保持使能，正常运行
    CMD_FAULT_RESET = 0x0080,        // 1000 0000：故障复位
};

/* 模式切换子状态 */
enum class ModeSwitchStep {
    IDLE,               // 空闲
    SET_MODE,           // 设置新模式
    CLEAR,              // 清除
    DISABLE,            // 失能
    ENABLE,             // 使能
    OPERATING,          // 开始运行
    DONE                // 完成
};


/* 定义交换数据结构 */
#pragma pack(push, 1)
/* TxPD0主站发送给从站 */
struct TxPDO
{
    uint16_t control_word;         // 0x6040:控制字
    int32_t  target_pos;           // 0x607A:目标位置/PVT期望位置
    int32_t  target_vel;           // 0x60FF:目标转速/PVT期望速度
    int16_t  target_torque;        // 0x6071:目标扭矩/PVT力矩前馈
    int32_t  pvt_kp;               // 0x2000:PVT_KP，实际Kp的1000倍
    int32_t  pvt_kd;               // 0x2001:PVT_KD，实际Kd的1000倍
    int8_t   op_mode;              // 0x6060:设置运行模式
    uint8_t  reserved;             // 0x2FFD (1字节填充)
}__attribute__((packed));
/* RxPD0主站接受从站数据 */   
struct RxPDO
{
    uint16_t status_word;     // 0x6041:反馈状态字
    int32_t pos;              // 0x6064:电机实际位置
    int32_t vel;              // 0x606C:电机实际转速
    int16_t torque;           // 0x6077:电机实际扭矩
    uint16_t error;           // 0x603F
    int8_t op_mode;           // 0x6061:显示运行模式
    uint8_t reserved;         // 0x2FFE (1字节填充)
}__attribute__((packed));
#pragma pack(pop)


static_assert(sizeof(TxPDO) == myactua::TX_PDO_SIZE);
static_assert(sizeof(RxPDO) == myactua::RX_PDO_SIZE);
}
