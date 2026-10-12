#pragma once
#include <vector>
#include <deque>
#include <memory>
#include <cstdio>
#include <thread>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <functional>
#include "MotorTypes.hpp"
#include "EthercatAdapter.hpp"
#include "ThreadSafeQueue.hpp"
#include "ControlTypes.hpp"

#define LIMIT(VAL, MIN, MAX) ((VAL)<(MIN)?(MIN):((VAL)>(MAX)?(MAX):(VAL)))

namespace myactua{


enum class MotorStep {
    IDLE,
    ENABLING,
    RUNNING,
    STOPPED,
    FAULT,
    MODE_SWITCHING
};


class MYACTUA {
public:
    /* 所有命令的本质都是操作这个结构体，在process_single_motor中进行下一步处理 */
    struct DesiredState {
        bool enabled;
        ControlMode mode;
        double setpoint;
        MitSetpoint mit_setpoint;

        DesiredState()
            : enabled(false), mode(ControlMode::CSP), setpoint(0.0),
              mit_setpoint() {}
    };

    struct ObservedState {
        bool fault;
        bool operation_enabled;
        bool switched_on;
        bool ready_to_switch_on;
        uint16_t status_word;
        uint16_t error_code;
        ControlMode mode;

        ObservedState()
            : fault(false), operation_enabled(false), switched_on(false),
              ready_to_switch_on(false), status_word(0), error_code(0),
              mode(ControlMode::CSP) {}
    };

    struct MotorState{
        int slave_index;

        DesiredState    desired;
        ObservedState   observed;

        MotorStep step;
        ModeSwitchStep mode_switch_step;

        TxPDO tx;
        RxPDO rx;

        bool comm_ok;
        bool prev_comm_ok;  // 上一周期的 comm_ok，用于检测 offline→online 切换
        uint32_t comm_offline_total_count;

        bool print_info;

        MotorState(int index)
        : slave_index(index), desired(), observed(), step(MotorStep::IDLE),
          mode_switch_step(ModeSwitchStep::IDLE), tx({}), rx({}), comm_ok(false),
          prev_comm_ok(false), comm_offline_total_count(0), print_info(false) {}
    };

    MYACTUA(std::shared_ptr<EthercatAdapter> adapter, int num_motors);

    ~MYACTUA();

    void send_command(const ControlCommand& cmd);   // 异步发送控制命令

    bool connect(const char* ifname);
    bool wait_all_slaves_ready(
        int timeout_ms = 30000,
        int poll_ms = 100,
        const std::function<bool()>& should_stop = {}) const;
    bool start();  // 启动实时线程；实时调度/亲和性验证失败时返回 false
    void shutdown();  // 关闭实时线程

    std::vector<MotorStatusSnapshot> get_status();  // 获取电机状态快照
    std::vector<double> get_joint_q_rad();          // 关节位置(rad)
    std::vector<double> get_joint_vel_rad_s();      // 关节速度(rad/s)
    std::vector<double> get_joint_tau_raw();        // 关节力矩(raw)

    using StatusCallback = std::function<void(const std::vector<MotorStatusSnapshot>&)>;  // 电机状态快照回调函数
    void set_status_callback(StatusCallback cb);  // 设置电机状态快照回调函数
    bool is_running() const { return running_; }  // 是否正在运行实时线程
    bool is_realtime_scheduling_ready() const noexcept {
        return rt_start_permitted_.load(std::memory_order_acquire);
    }

    /* 只打印存在于 slave_indices 中ID的电机信息 */
    void set_print_info(const std::vector<int>& slave_indices);
    void print_motors_info(void);

    static double raw_pos_to_rad(double raw_pos);
    static double rad_to_deg(double rad);
    static double raw_vel_to_rad_s(double raw_vel);

    
private:
    std::shared_ptr<EthercatAdapter> _adapter;
    std::vector<MotorState> _motors;

    /* 控制命令队列 */
    ThreadSafeQueue<ControlCommand> cmd_queue_;
    std::mutex status_mutex_;

    /* 电机的离散命令队列 */
    std::vector<std::deque<DiscreteCommand>> discrete_cmd_queues_;
    uint64_t discrete_cmd_tick_{0};  // 离散命令时间戳（按控制周期递增）

    static constexpr uint32_t kCommWatchdogFaultCycles = 10;
    static constexpr uint32_t kCommWatchdogRecoveryCycles = 10;
    // 认定链路"真正稳定"所需的连续完整周期数（1 kHz 下约 2 秒）。
    // 用于区分启动假象与真实稳定：wait_all_slaves_ready() 返回的瞬间 domain
    // 状态可能恰好显示完整，随后 working counter 又回落。
    static constexpr uint32_t kStableHealthyCycles = 2000;

    std::thread rt_thread_;
    std::atomic<bool> running_{false};  // 实时线程运行标志
    std::atomic<bool> rt_start_permitted_{false};

    std::atomic<bool> whole_body_fault_latched_{false};
    // PDO 交换是否已确认稳定（连续 kStableHealthyCycles 个周期完整）。
    // 确认之前不启用通信看门狗：从站逐个进入 OP 并开始 PDO 交换可能耗时
    // 数十秒，属正常暂态；此时若按故障锁存，会连带丢弃上层下发的 SET_MODE，
    // 使驱动停在默认 CSP 而上层按 MIT/PVT 发目标。确认之后看门狗严格生效。
    bool ever_process_data_ok_{false};
    uint32_t healthy_streak_{0};   // 连续完整周期计数，一旦不完整立即清零
    bool restart_all_requested_{false};
    uint32_t process_data_fail_count_{0};
    uint32_t recovery_healthy_count_{0};
    uint64_t fault_tick_{0};

    std::vector<MotorStatusSnapshot> status_snapshot_;

    std::vector<int> print_motor_ids_;
    
    StatusCallback status_callback_;

    void rt_thread_func();
    void process_commands();
    void update(const std::vector<double>& setvalues);
    void update_status_snapshot();

    /* 处理离散命令相关 */
    void enqueue_discrete_command(const ControlCommand& cmd);
    void service_discrete_commands(const std::vector<bool>& comm_ok);
    void apply_discrete_command_to_motor(int motor_index, const DiscreteCommand& cmd);
    bool is_discrete_command_satisfied(const MotorState& motor, const DiscreteCommand& cmd) const;


    void updateCommunicationWatchdog(
        bool process_data_ok, const EthercatBusHealthSnapshot& health);
    void latchCommunicationFault(const EthercatBusHealthSnapshot& health);
    void clearCommunicationFault();
    void applyWholeBodyQuickStop();
    void resetMotorTargetsToFeedback(MotorState& motor);

    void refresh_observed_state(MotorState& motor);

    void process_single_motor(MotorState& motor, double setvalue);

    void handle_mode_switching(MotorState& motor);

    ControlWordCommand get_next_control_word(uint16_t status_word);
};

} // namespace myactua
