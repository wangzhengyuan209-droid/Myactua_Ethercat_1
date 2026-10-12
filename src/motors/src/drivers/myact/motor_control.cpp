#include "motor_control.hpp"
#include "motor_units.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <thread>
#include <cstring>
#include <time.h>
#include <pthread.h>
#include <sched.h>

namespace myactua{

#define NSEC_PER_SEC (1000000000L)
#define CLOCK_TO_USE CLOCK_MONOTONIC

namespace {
constexpr double kPi = myactua::motor_units::kPi;
constexpr double kRawPosToRad = myactua::motor_units::kRawPosToRad;
constexpr double kRawVelToRadPerSec = myactua::motor_units::kRawVelToRadPerSec;
constexpr double kRadToDeg = 180.0 / kPi;

bool is_18bit_ankle_motor(int motor_index)
{
    return myactua::motor_units::is_ankle_motor(motor_index);
}

double position_raw_scale(int motor_index)
{
    return is_18bit_ankle_motor(motor_index) ? 2.0 : 1.0;
}

double raw_pos_to_rad_for_motor(double raw, int motor_index)
{
    return myactua::motor_units::raw_pos_to_rad(raw, motor_index);
}

double raw_vel_to_rad_s_for_motor(double raw, int motor_index)
{
    return myactua::motor_units::raw_vel_to_rad_s(raw, motor_index);
}

double rad_s_to_raw_for_motor(double velocity_rad_s, int motor_index)
{
    return myactua::motor_units::rad_s_to_raw_vel(velocity_rad_s, motor_index);
}

int32_t double_to_i32(double value)
{
    if (!std::isfinite(value)) {
        return 0;
    }
    const double lo = static_cast<double>(std::numeric_limits<int32_t>::min());
    const double hi = static_cast<double>(std::numeric_limits<int32_t>::max());
    return static_cast<int32_t>(std::llround(std::max(lo, std::min(hi, value))));
}

int16_t double_to_i16(double value)
{
    if (!std::isfinite(value)) {
        return 0;
    }
    const double lo = static_cast<double>(std::numeric_limits<int16_t>::min());
    const double hi = static_cast<double>(std::numeric_limits<int16_t>::max());
    return static_cast<int16_t>(std::llround(std::max(lo, std::min(hi, value))));
}
}

/* 电机控制器构造函数 */
MYACTUA::MYACTUA(std::shared_ptr<EthercatAdapter> adapter, int num_motors) : _adapter(adapter)
{
    /* 初始化电机状态列表 */
    for (int i = 0; i < num_motors; i++) {
        _motors.emplace_back(i);
        print_motor_ids_.push_back(i);
    }
    status_snapshot_.resize(num_motors);
    discrete_cmd_queues_.resize(num_motors);
}

/* 析构函数 */
MYACTUA::~MYACTUA()
{
    shutdown();
}

/* 连接网卡函数 */
bool MYACTUA::connect(const char* ifname)
{
    if(_adapter->init(ifname))
        return true;
    return false;
}


/* 等待所有 EtherCAT 从站进入 OP */
bool MYACTUA::wait_all_slaves_ready(
    int timeout_ms,
    int poll_ms,
    const std::function<bool()>& should_stop) const
{
    if (!_adapter) {
        std::cerr << "[MYACTUA] EtherCAT adapter is null." << std::endl;
        return false;
    }

    using Clock = std::chrono::steady_clock;
    const int effective_timeout_ms = std::max(0, timeout_ms);
    const int effective_poll_ms = std::max(1, poll_ms);
    const auto start_time = Clock::now();
    const auto deadline = start_time + std::chrono::milliseconds(effective_timeout_ms);
    auto next_log_time = start_time;
    bool first_check = true;

    while (first_check || (effective_timeout_ms > 0 && Clock::now() < deadline)) {
        first_check = false;

        if (should_stop && should_stop()) {
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                Clock::now() - start_time).count();
            std::cout << "[MYACTUA] wait_all_slaves_ready interrupted after "
                      << elapsed_ms << " ms" << std::endl;
            return false;
        }

        _adapter->receivePhysical();
        _adapter->sendPhysical();

        int ready_count = 0;
        for (int i = 0; i < static_cast<int>(_motors.size()); ++i) {
            if (_adapter->isConfigured(i)) {
                ++ready_count;
            }
        }

        if (ready_count == static_cast<int>(_motors.size())) {
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                Clock::now() - start_time).count();
            std::cout << "[MYACTUA] All slaves ready in "
                      << elapsed_ms << " ms" << std::endl;
            return true;
        }

        const auto now = Clock::now();
        if (now >= next_log_time) {
            std::cout << "[MYACTUA] EtherCAT ready: "
                      << ready_count << "/" << _motors.size() << std::endl;
            next_log_time = now + std::chrono::milliseconds(effective_poll_ms);
        }

        if (effective_timeout_ms == 0) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - start_time).count();
    std::cout << "[MYACTUA] wait_all_slaves_ready timeout after "
              << elapsed_ms << " ms" << std::endl;
    return false;
}


/* 电机实时控制线程 */
void MYACTUA::rt_thread_func()
{
    while (running_ && !rt_start_permitted_.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    if (!running_) {
        return;
    }

    struct timespec next_period;
    clock_gettime(CLOCK_TO_USE, &next_period);
    const long period_ns = 1000000;

    while (running_) {
        // 1. 收取本周期 EtherCAT 输入
        _adapter->receivePhysical();
        // 2.处理控制命令
        process_commands();
        // 3.执行控制周期
        update({});
        // 4. 发送本周期 EtherCAT 输出
        _adapter->sendPhysical();
        // 5.更新电机状态快照
        update_status_snapshot();
        // 6.调用回调函数
        if (status_callback_) {
            status_callback_(status_snapshot_);
        }
        // 7.等待下一个周期
        next_period.tv_nsec += period_ns;
        if (next_period.tv_nsec >= NSEC_PER_SEC) {
            next_period.tv_nsec -= NSEC_PER_SEC;
            next_period.tv_sec++;
        }
        clock_nanosleep(CLOCK_TO_USE, TIMER_ABSTIME, &next_period, nullptr);
    }
}


void MYACTUA::process_commands()
{
    ControlCommand cmd;
    while (cmd_queue_.pop(cmd, 0)) {
        if (whole_body_fault_latched_.load(std::memory_order_acquire)) {
            if (cmd.kind == ControlCommandKind::DISCRETE &&
                cmd.discrete_type == DiscreteCommandType::RESTART &&
                cmd.slave_index == ControlCommand::kAllSlaves) {
                restart_all_requested_ = true;
                recovery_healthy_count_ = 0;
            } else if (cmd.kind == ControlCommandKind::DISCRETE &&
                       cmd.discrete_type == DiscreteCommandType::STOP) {
                // STOP remains an accepted safety command, but never clears the latch.
                if (cmd.slave_index == ControlCommand::kAllSlaves) {
                    applyWholeBodyQuickStop();
                } else if (cmd.slave_index >= 0 &&
                           cmd.slave_index < static_cast<int>(_motors.size())) {
                    auto& motor = _motors[static_cast<std::size_t>(cmd.slave_index)];
                    motor.desired.enabled = false;
                    motor.tx.control_word = CMD_QUICK_STOP;
                    motor.tx.target_pos = motor.rx.pos;
                    motor.tx.target_vel = 0;
                    motor.tx.target_torque = 0;
                    motor.tx.pvt_kp = 0;
                    motor.tx.pvt_kd = 0;
                    _adapter->send(motor.slave_index, motor.tx);
                }
            }
            // 故障锁存期间只接受 STOP 和全机 RESTART；其余命令全部丢弃。
            continue;
        }

        if (cmd.kind == ControlCommandKind::DISCRETE) {
            enqueue_discrete_command(cmd);
            continue;
        }

        switch (cmd.setpoint_type) {
            /* 设置 DesiredState 中的标量目标值 */
            case SetpointCommandType::SCALAR_SETPOINTS:
                if (cmd.slave_index == ControlCommand::kAllSlaves) {
                    for (size_t i = 0; i < cmd.scalar_setpoints.size() && i < _motors.size(); i++) {
                        _motors[i].desired.setpoint = cmd.scalar_setpoints[i];
                    }
                } else if (cmd.slave_index >= 0 &&
                           cmd.slave_index < static_cast<int>(_motors.size()) &&
                           !cmd.scalar_setpoints.empty()) {
                    _motors[cmd.slave_index].desired.setpoint = cmd.scalar_setpoints.front();
                }
                break;

            case SetpointCommandType::MIT_SETPOINTS:
                if (cmd.slave_index == ControlCommand::kAllSlaves) {
                    for (size_t i = 0; i < cmd.mit_setpoints.size() && i < _motors.size(); i++) {
                        _motors[i].desired.mit_setpoint = cmd.mit_setpoints[i];
                    }
                } else if (cmd.slave_index >= 0 &&
                           cmd.slave_index < static_cast<int>(_motors.size()) &&
                           !cmd.mit_setpoints.empty()) {
                    _motors[cmd.slave_index].desired.mit_setpoint = cmd.mit_setpoints.front();
                }
                break;
        }
    }
}

/* 如果为离散命令需要确保完整执行； */
void MYACTUA::enqueue_discrete_command(const ControlCommand& cmd)
{
    /* 就地定义、使用，拉满封装性 */
    auto enqueue_one = [this, &cmd](int idx) {
        if (idx < 0 || idx >= static_cast<int>(_motors.size())) {
            return;
        }
        DiscreteCommand pending(cmd.discrete_type, cmd.mode);
        pending.phase = DiscretePhase::QUEUED;  // 初始状态为 QUEUED(入队列)

        pending.enqueue_tick      = discrete_cmd_tick_;
        pending.next_retry_tick   = discrete_cmd_tick_;  // 首次 apply 不等待
        pending.next_verify_tick  = discrete_cmd_tick_;  // 占位值 
        pending.deadline_tick     = discrete_cmd_tick_ + kDiscreteTimeoutTicks;

        pending.cur_retry = 0;
        pending.max_retries = kDiscreteMaxRetries;
        pending.stable_success_cycles = 0;
        pending.fail_reason = kDiscreteFailNone;

        discrete_cmd_queues_[idx].push_back(pending);
    };

    /* kAllSlaves 表示对所有电机执行命令 */
    if (cmd.slave_index == ControlCommand::kAllSlaves) {
        for (int i = 0; i < static_cast<int>(_motors.size()); ++i) {
            if (cmd.discrete_type == DiscreteCommandType::STOP) {
                discrete_cmd_queues_[i].clear();
            }
            enqueue_one(i);
        }
    } else {
        if (cmd.discrete_type == DiscreteCommandType::STOP &&
            cmd.slave_index >= 0 &&
            cmd.slave_index < static_cast<int>(_motors.size())) {
            discrete_cmd_queues_[static_cast<std::size_t>(cmd.slave_index)].clear();
        }
        enqueue_one(cmd.slave_index);
    }
}


void MYACTUA::update(const std::vector<double> &setvalues)
{
    static int print_count = 0;
    ++discrete_cmd_tick_;  // 离散命令时间戳增加
    std::vector<bool> comm_ok(_motors.size(), false);

    /* 接受电机回传数据，并记录当前周期通信状态 */
    for (size_t i = 0; i < _motors.size(); i++)
    {
        comm_ok[i] = _adapter->isConfigured(_motors[i].slave_index);
        if (!comm_ok[i]) {
            ++_motors[i].comm_offline_total_count;
        }
        _motors[i].comm_ok = comm_ok[i];

        /* 只接受通信正常的电机数据 */
        if (comm_ok[i]) {
            _motors[i].rx = _adapter->receive(_motors[i].slave_index);
            refresh_observed_state(_motors[i]);
        }

        _motors[i].prev_comm_ok = comm_ok[i];
    }

    const EthercatBusHealthSnapshot health = _adapter->getBusHealth();
    const bool process_data_ok =
        health.master_link_up && health.wc_state == EC_WC_COMPLETE;
    const bool all_motors_healthy =
        process_data_ok && std::all_of(comm_ok.begin(), comm_ok.end(),
                                       [](bool ok) { return ok; });
    updateCommunicationWatchdog(all_motors_healthy, health);

    if (whole_body_fault_latched_.load(std::memory_order_acquire)) {
        applyWholeBodyQuickStop();
        return;
    }

    /* 自动恢复已禁用：任何通信边沿都必须由上层显式全机 RESTART 处理。 */
    /* 处理离散命令 */
    service_discrete_commands(comm_ok);

    /* 设置电机目标值。通信异常时保持当前控制状态，不覆写 step。 */
    for (size_t i = 0; i < _motors.size(); i++)
    {
        if (!comm_ok[i]) {
            continue;
        }
        double val = (i < setvalues.size()) ? setvalues[i] : _motors[i].desired.setpoint;
        process_single_motor(_motors[i], val);
    }

    /* 最终发送 */
    for (size_t i = 0; i < _motors.size(); i++)
    {
        if (!comm_ok[i]) continue;
        _adapter->send(_motors[i].slave_index, _motors[i].tx);
    }

    if (++print_count >= 500) {
        print_count = 0;
        if (!print_motor_ids_.empty()) {
            print_motors_info();
        }
    }
}


void MYACTUA::updateCommunicationWatchdog(
    bool process_data_ok, const EthercatBusHealthSnapshot& health)
{
    if (whole_body_fault_latched_.load(std::memory_order_acquire)) {
        if (!restart_all_requested_) {
            recovery_healthy_count_ = 0;
            return;
        }

        if (!process_data_ok) {
            recovery_healthy_count_ = 0;
            return;
        }

        ++recovery_healthy_count_;
        if (recovery_healthy_count_ >= kCommWatchdogRecoveryCycles) {
            clearCommunicationFault();
            // 仅在健康窗口完成后启动显式全机恢复；故障前的命令已被丢弃。
            enqueue_discrete_command(ControlCommand::Restart());
        }
        return;
    }

    if (process_data_ok) {
        process_data_fail_count_ = 0;
        // 必须连续完整足够久才算"真正稳定"：wait_all_slaves_ready() 返回的瞬间
        // domain 状态可能恰好显示完整，随后 working counter 立刻回落，属启动假象。
        if (!ever_process_data_ok_ && ++healthy_streak_ >= kStableHealthyCycles) {
            ever_process_data_ok_ = true;
        }
        return;
    }

    // 尚未确认稳定前不判故障：从站逐个进入 OP 并开始 PDO 交换耗时可达数十秒，
    // 期间 working counter 偏低属正常暂态。
    if (!ever_process_data_ok_) {
        healthy_streak_ = 0;
        process_data_fail_count_ = 0;
        return;
    }

    ++process_data_fail_count_;
    if (process_data_fail_count_ >= kCommWatchdogFaultCycles) {
        latchCommunicationFault(health);
    }
}

void MYACTUA::latchCommunicationFault(const EthercatBusHealthSnapshot& health)
{
    if (whole_body_fault_latched_.exchange(true, std::memory_order_acq_rel)) {
        return;
    }

    restart_all_requested_ = false;
    recovery_healthy_count_ = 0;
    fault_tick_ = discrete_cmd_tick_;
    for (auto& queue : discrete_cmd_queues_) {
        queue.clear();
    }
    cmd_queue_.clear();

    std::cerr << "[MYACTUA] EtherCAT process-data fault latched at tick "
              << fault_tick_ << ": link_up=" << (health.master_link_up ? 1 : 0)
              << ", wc_state=" << static_cast<int>(health.wc_state)
              << ", working_counter=" << health.working_counter
              << ", ever_ok=" << (ever_process_data_ok_ ? 1 : 0)
              << ", fail_count=" << process_data_fail_count_ << std::endl;
}

void MYACTUA::clearCommunicationFault()
{
    for (auto& queue : discrete_cmd_queues_) {
        queue.clear();
    }

    for (auto& motor : _motors) {
        motor.desired.enabled = false;
        motor.mode_switch_step = ModeSwitchStep::IDLE;
        resetMotorTargetsToFeedback(motor);
    }

    process_data_fail_count_ = 0;
    recovery_healthy_count_ = 0;
    restart_all_requested_ = false;
    whole_body_fault_latched_.store(false, std::memory_order_release);

    std::cout << "[MYACTUA] EtherCAT process-data fault cleared after explicit "
                 "all-motor restart and healthy recovery." << std::endl;
}

void MYACTUA::applyWholeBodyQuickStop()
{
    for (auto& motor : _motors) {
        motor.step = MotorStep::FAULT;
        motor.desired.enabled = false;
        motor.mode_switch_step = ModeSwitchStep::IDLE;
        motor.tx.control_word = CMD_QUICK_STOP;
        motor.tx.op_mode = motor.desired.mode;
        motor.tx.target_pos = motor.rx.pos;
        motor.tx.target_vel = 0;
        motor.tx.target_torque = 0;
        motor.tx.pvt_kp = 0;
        motor.tx.pvt_kd = 0;
        _adapter->send(motor.slave_index, motor.tx);
    }
}

void MYACTUA::resetMotorTargetsToFeedback(MotorState& motor)
{
    motor.desired.setpoint =
        raw_pos_to_rad_for_motor(motor.rx.pos, motor.slave_index) * kRadToDeg;
    motor.desired.mit_setpoint = MitSetpoint(
        motor.desired.setpoint, 0.0, 0.0, 0.0, 0.0);
    motor.tx.control_word = CMD_DISABLE_OPERATION;
    motor.tx.op_mode = motor.desired.mode;
    motor.tx.target_pos = motor.rx.pos;
    motor.tx.target_vel = 0;
    motor.tx.target_torque = 0;
    motor.tx.pvt_kp = 0;
    motor.tx.pvt_kd = 0;
}


void MYACTUA::refresh_observed_state(MotorState& motor)
{
    const uint16_t sw = motor.rx.status_word;
    motor.observed.status_word = sw;
    motor.observed.error_code = motor.rx.error;
    motor.observed.mode = (ControlMode)motor.rx.op_mode;
    motor.observed.ready_to_switch_on = is_ready_to_switch_on(sw);
    motor.observed.switched_on = is_switched_on(sw);
    motor.observed.operation_enabled = is_operation_enabled(sw);
    motor.observed.fault = is_fault(sw);
}


void MYACTUA::service_discrete_commands(const std::vector<bool>& comm_ok)
{
    for (size_t i = 0; i < _motors.size(); ++i) {
        auto& queue = discrete_cmd_queues_[i];
        /* 无命令则跳过 */
        if (queue.empty()) {
            continue;
        }
        /* 获取命令和对应的电机 */
        auto& cmd   = queue.front();
        auto& motor = _motors[i];

        /* 当前命令已完成 */
        if (cmd.phase == DiscretePhase::DONE) {
            queue.pop_front();
            continue;
        }
        if (cmd.phase == DiscretePhase::FAILED) {
            std::cerr << "[MYACTUA] discrete command failed on motor " << i
                      << ", type=" << static_cast<int>(cmd.type)
                      << ", reason=" << cmd.fail_reason
                      << ", retry=" << cmd.cur_retry << "\n";
            queue.pop_front();
            continue;
        }

        /* 超时 */
        if (discrete_cmd_tick_ > cmd.deadline_tick) {
            cmd.phase = DiscretePhase::FAILED;
            cmd.fail_reason = kDiscreteFailTimeout;
            continue;
        }

        /* 当前掉线，保持等待，不推进命令阶段 */
        if (!comm_ok[i]) {
            cmd.stable_success_cycles = 0;
            continue;
        }

        /* 驱动明确报错，命令直接失败（RESTART 除外 — 让 process_single_motor 发 FAULT RESET） */
        if (motor.observed.fault) {
            if (cmd.type == DiscreteCommandType::RESTART) {
                cmd.stable_success_cycles = 0;
                continue;
            }
            cmd.phase = DiscretePhase::FAILED;
            cmd.fail_reason = kDiscreteFailFault;
            continue;
        }

        if (cmd.phase == DiscretePhase::QUEUED) {
            cmd.phase = DiscretePhase::APPLY_PENDING;
        }

        if (cmd.phase == DiscretePhase::APPLY_PENDING) {
            if (cmd.cur_retry >= cmd.max_retries) {
                cmd.phase = DiscretePhase::FAILED;
                cmd.fail_reason = kDiscreteFailMaxRetry;
                continue;
            }
            if (discrete_cmd_tick_ >= cmd.next_retry_tick) {
                apply_discrete_command_to_motor(static_cast<int>(i), cmd);
                cmd.cur_retry += 1;
                cmd.next_retry_tick  = discrete_cmd_tick_ + kDiscreteRetryTicks;
                cmd.next_verify_tick = discrete_cmd_tick_ + kDiscreteVerifyIntervalTicks;
                cmd.stable_success_cycles = 0;
                cmd.phase = DiscretePhase::VERIFYING;
            }
            continue;
        }

        if (cmd.phase == DiscretePhase::VERIFYING) {
            /* 没到验证时间 */
            if (discrete_cmd_tick_ < cmd.next_verify_tick) {
                continue;
            }

            /* 验证成功 */
            if (is_discrete_command_satisfied(motor, cmd)) {
                cmd.stable_success_cycles += 1;
                if (cmd.stable_success_cycles >= kDiscreteSuccessStableTicks) {
                    cmd.phase = DiscretePhase::DONE;
                    queue.pop_front();
                } else {
                    cmd.next_verify_tick = discrete_cmd_tick_ + kDiscreteVerifyIntervalTicks;
                }
            }
            /* 验证失败 */
            else {
                cmd.stable_success_cycles = 0;
                /* 重发周期到了，可以进行重发 */
                if (discrete_cmd_tick_ >= cmd.next_retry_tick) {
                    cmd.phase = DiscretePhase::APPLY_PENDING;
                }
            }
        }
    }
}

/* 设置 DesiredState 结构体中的数据 */
void MYACTUA::apply_discrete_command_to_motor(int motor_index, const DiscreteCommand& cmd)
{
    if (motor_index < 0 || motor_index >= static_cast<int>(_motors.size())) {
        return;
    }

    MotorState& motor = _motors[motor_index];
    switch (cmd.type) {
        case DiscreteCommandType::STOP:
            // Edge-triggered: avoid resetting mode-switch state on retries.
            if (motor.desired.enabled) {
                motor.desired.enabled = false;
                motor.mode_switch_step = ModeSwitchStep::IDLE;
            }
            break;
        case DiscreteCommandType::RESTART:
            // Edge-triggered: first restart arms enable flow; later retries are no-op.
            if (!motor.desired.enabled) {
                motor.desired.enabled = true;
                motor.mode_switch_step = ModeSwitchStep::IDLE;
            }
            break;
        case DiscreteCommandType::SET_MODE:
            if (motor.desired.mode != cmd.mode) {
                motor.desired.mode  = cmd.mode;
                motor.mode_switch_step = ModeSwitchStep::IDLE;
            }
            break;
    }
}


bool MYACTUA::is_discrete_command_satisfied(const MotorState& motor, const DiscreteCommand& cmd) const
{
    const auto& observed = motor.observed;
    if (observed.fault) {
        return false;
    }

    switch (cmd.type) {
        case DiscreteCommandType::STOP:
            return !observed.operation_enabled;
        case DiscreteCommandType::RESTART:
            return  observed.operation_enabled;
        case DiscreteCommandType::SET_MODE:
            return observed.mode == cmd.mode;
    }
    return false;
}



void MYACTUA::process_single_motor(MotorState& motor, double setvalue)
{
    const auto& observed = motor.observed;
    const auto& desired  = motor.desired;
    const uint16_t sw = observed.status_word;

    if (observed.fault) {
        motor.step = MotorStep::FAULT;
        motor.mode_switch_step = ModeSwitchStep::IDLE;
        motor.tx.op_mode = desired.mode;
        motor.tx.target_pos = motor.rx.pos;
        motor.tx.target_vel = 0;
        motor.tx.target_torque = 0;
        motor.tx.pvt_kp = 0;
        motor.tx.pvt_kd = 0;
        // desired.enabled ? 发 FAULT RESET (0x80) 让驱动退出 FAULT 态
        //                 : 不使能则保持 shutdown
        motor.tx.control_word = desired.enabled ? CMD_FAULT_RESET : CMD_SHUTDOWN;
        return;
    }

    /* 需要停止 */
    if (!desired.enabled) {
        motor.step = MotorStep::STOPPED;
        motor.mode_switch_step = ModeSwitchStep::IDLE;
        motor.tx.control_word = CMD_DISABLE_OPERATION;
        motor.tx.op_mode = desired.mode;
        motor.tx.target_vel = 0;
        motor.tx.target_pos = motor.rx.pos;
        motor.tx.target_torque = 0;
        motor.tx.pvt_kp = 0;
        motor.tx.pvt_kd = 0;
        return;
    }

    if (desired.mode == ControlMode::NONE) {
        motor.step = MotorStep::IDLE;
        motor.mode_switch_step = ModeSwitchStep::IDLE;
        motor.tx.control_word = CMD_SHUTDOWN;
        motor.tx.op_mode = ControlMode::NONE;
        motor.tx.target_pos = motor.rx.pos;
        motor.tx.target_vel = 0;
        motor.tx.target_torque = 0;
        motor.tx.pvt_kp = 0;
        motor.tx.pvt_kd = 0;
        return;
    }

    const bool mode_switch_active =
         (observed.mode != desired.mode) || (motor.mode_switch_step != ModeSwitchStep::IDLE);
    if (mode_switch_active) {
        motor.step = MotorStep::MODE_SWITCHING;
        handle_mode_switching(motor);
        // 模式切换采用闭环状态机，本周期不进入常规运行逻辑，避免覆盖切换命令。
        return;
    }

    /* 模式一致后进入使能/运行控制 */
    motor.tx.control_word = get_next_control_word(sw);
    motor.tx.op_mode = desired.mode;

    if (observed.operation_enabled) {
        motor.step = MotorStep::RUNNING;
        motor.tx.pvt_kp = 0;
        motor.tx.pvt_kd = 0;
        switch (desired.mode) {
            case ControlMode::CSV:  // 速度命令：rad/s → 原始速度单位
                motor.tx.target_vel = double_to_i32(
                    rad_s_to_raw_for_motor(setvalue, motor.slave_index));
                break;
            case ControlMode::CSP:  // setvalue: deg, 1 rev = 131072 plus
                motor.tx.target_pos = double_to_i32(
                    myactua::motor_units::rad_to_raw_pos(
                        setvalue / kRadToDeg, motor.slave_index));
                break;
            case ControlMode::CST:  // 电流百分比
                motor.tx.target_torque = static_cast<int16_t>(setvalue);
                break;
            case ControlMode::PVT: {
                const MitSetpoint& mit = desired.mit_setpoint;
                motor.tx.target_pos = double_to_i32(
                    myactua::motor_units::rad_to_raw_pos(
                        mit.position_deg / kRadToDeg, motor.slave_index));
                motor.tx.target_vel = double_to_i32(
                    rad_s_to_raw_for_motor(mit.velocity_rad_s, motor.slave_index));
                motor.tx.target_torque = double_to_i16(mit.torque_ff_permille);
                motor.tx.pvt_kp = double_to_i32(mit.kp * 1000.0);
                motor.tx.pvt_kd = double_to_i32(mit.kd * 1000.0);
                break;
            }
            default:
                break;
        }
    } else {
        motor.step = MotorStep::ENABLING;
        motor.tx.target_pos = motor.rx.pos;
        motor.tx.target_vel = 0;
        motor.tx.target_torque = 0;
        motor.tx.pvt_kp = 0;
        motor.tx.pvt_kd = 0;
    }
}


/* 状态机切换电机控制模式 */
void MYACTUA::handle_mode_switching(MotorState& motor)
{
    const auto& observed = motor.observed;
    const auto& desired  = motor.desired;
    const uint16_t sw = observed.status_word;
    const bool mode_ok = (observed.mode == desired.mode);

    // 模式切换期间固定目标，避免切换过程中产生突变。
    motor.tx.op_mode = desired.mode;
    motor.tx.target_pos = motor.rx.pos;
    motor.tx.target_vel = 0;
    motor.tx.target_torque = 0;
    motor.tx.pvt_kp = 0;
    motor.tx.pvt_kd = 0;

    switch (motor.mode_switch_step)
    {
        case ModeSwitchStep::IDLE:
            // 1) 写 0x6060(目标模式)
            motor.tx.control_word = CMD_SHUTDOWN;
            motor.mode_switch_step = ModeSwitchStep::SET_MODE;
            break;

        case ModeSwitchStep::SET_MODE:
            // 保持 0x6040=6，在失能态等待 0x6061 回读到目标模式，闭环推进。
            motor.tx.control_word = CMD_SHUTDOWN;
            if (mode_ok) {
                motor.mode_switch_step = ModeSwitchStep::CLEAR;
            }
            break;

        case ModeSwitchStep::CLEAR:
            // 2) 读 0x6064，并将 0x607A 对齐到当前位置。
            // 对齐命令至少发送一个周期后再进入下一步。
            motor.tx.control_word = CMD_SHUTDOWN;
            if (!mode_ok) {
                motor.mode_switch_step = ModeSwitchStep::SET_MODE;
            } else {
                motor.mode_switch_step = ModeSwitchStep::DISABLE;
            }
            break;

        case ModeSwitchStep::DISABLE:
            // 3-1) 写 0x6040=6，等待状态字进入 Ready to switch on。
            motor.tx.control_word = CMD_SHUTDOWN;
            if (!mode_ok) {
                motor.mode_switch_step = ModeSwitchStep::SET_MODE;
                break;
            }
            if (is_ready_to_switch_on(sw) && !is_switched_on(sw) && !is_operation_enabled(sw)) {
                motor.mode_switch_step = ModeSwitchStep::ENABLE;
            }
            break;

        case ModeSwitchStep::ENABLE:
            // 3-2) 写 0x6040=7，等待状态字进入 Switched on。
            motor.tx.control_word = CMD_SWITCH_ON;
            if (!mode_ok) {
                motor.mode_switch_step = ModeSwitchStep::SET_MODE;
                break;
            }
            if (is_switched_on(sw) && !is_operation_enabled(sw)) {
                motor.mode_switch_step = ModeSwitchStep::OPERATING;
            } else if (!is_ready_to_switch_on(sw)) {
                motor.mode_switch_step = ModeSwitchStep::DISABLE;
            }
            break;

        case ModeSwitchStep::OPERATING:
            // 3-3) 写 0x6040=15，等待状态字进入 Operation enabled。
            motor.tx.control_word = CMD_ENABLE_OPERATION;
            if (!mode_ok) {
                motor.mode_switch_step = ModeSwitchStep::SET_MODE;
                break;
            }
            if (is_operation_enabled(sw)) {
                motor.mode_switch_step = ModeSwitchStep::DONE;
            } else if (!is_switched_on(sw)) {
                motor.mode_switch_step = ModeSwitchStep::ENABLE;
            }
            break;

        case ModeSwitchStep::DONE:
            // 4) 闭环完成: 模式正确且已使能，切回常规运行控制。
            motor.tx.control_word = CMD_ENABLE_OPERATION;
            if (mode_ok && is_operation_enabled(sw)) {
                motor.mode_switch_step = ModeSwitchStep::IDLE;
                motor.step = MotorStep::RUNNING;
            } else {
                motor.mode_switch_step = mode_ok ? ModeSwitchStep::OPERATING : ModeSwitchStep::SET_MODE;
            }
            break;
    }
}


/* 依据当前状态获取下一个控制命令 */
ControlWordCommand MYACTUA::get_next_control_word(uint16_t status_word)
{
    if (!is_switched_on(status_word) && !is_ready_to_switch_on(status_word)) {
        return CMD_SHUTDOWN;
    }
    if (!is_switched_on(status_word) &&  is_ready_to_switch_on(status_word)) {
        return CMD_SWITCH_ON;
    }
    if (is_switched_on(status_word) && !is_operation_enabled(status_word)) {
        return CMD_ENABLE_OPERATION;
    }

    return CMD_ENABLE_OPERATION;
}



void MYACTUA::update_status_snapshot()
{
    std::lock_guard<std::mutex> lock(status_mutex_);
    for (size_t i = 0; i < _motors.size(); i++) {
        const auto& m = _motors[i];
        status_snapshot_[i].slave_index = m.slave_index;
        status_snapshot_[i].position = static_cast<double>(m.rx.pos);
        status_snapshot_[i].velocity = static_cast<double>(m.rx.vel);
        status_snapshot_[i].torque   = static_cast<double>(m.rx.torque);
        status_snapshot_[i].comm_ok  = m.comm_ok;
        status_snapshot_[i].status_word = m.rx.status_word;
        status_snapshot_[i].error_code  = m.rx.error;
        status_snapshot_[i].op_mode = (ControlMode)m.rx.op_mode;
        status_snapshot_[i].target_mode = m.desired.mode;
    }
}


double MYACTUA::raw_pos_to_rad(double raw_pos)
{
    return raw_pos * kRawPosToRad;
}


double MYACTUA::rad_to_deg(double rad)
{
    return rad * kRadToDeg;
}


double MYACTUA::raw_vel_to_rad_s(double raw_vel)
{
    return raw_vel * kRawVelToRadPerSec;
}


void MYACTUA::send_command(const ControlCommand& cmd)
{
    cmd_queue_.push(cmd);
}


std::vector<MotorStatusSnapshot> MYACTUA::get_status()
{
    std::lock_guard<std::mutex> lock(status_mutex_);
    return status_snapshot_;
}


std::vector<double> MYACTUA::get_joint_q_rad()
{
    const auto status = get_status();
    std::vector<double> q(status.size(), 0.0);
    for (std::size_t i = 0; i < status.size(); ++i) {
        q[i] = raw_pos_to_rad_for_motor(status[i].position,
                                        status[i].slave_index);
    }
    return q;
}


std::vector<double> MYACTUA::get_joint_vel_rad_s()
{
    const auto status = get_status();
    std::vector<double> dq(status.size(), 0.0);
    for (std::size_t i = 0; i < status.size(); ++i) {
        dq[i] = raw_vel_to_rad_s_for_motor(status[i].velocity,
                                           status[i].slave_index);
    }
    return dq;
}


std::vector<double> MYACTUA::get_joint_tau_raw()
{
    const auto status = get_status();
    std::vector<double> tau(status.size(), 0.0);
    for (std::size_t i = 0; i < status.size(); ++i) {
        tau[i] = status[i].torque;
    }
    return tau;
}


void MYACTUA::set_status_callback(StatusCallback cb)
{
    status_callback_ = std::move(cb);
}


/* 默认为不打印电机信息 */
void MYACTUA::set_print_info(const std::vector<int>& slave_indices)
{
    for (auto& motor : _motors) {
        motor.print_info = false;
    }
    print_motor_ids_.clear();

    for (int slave_index : slave_indices) {
        if (slave_index < 0 || slave_index >= static_cast<int>(_motors.size())) {
            continue;
        }
        /* 去重 */
        if (std::find(print_motor_ids_.begin(), print_motor_ids_.end(), slave_index) == print_motor_ids_.end()) {
            print_motor_ids_.push_back(slave_index);
            _motors[slave_index].print_info = true;
        }
    }
}


void MYACTUA::print_motors_info(void){
    printf("\033[2J\033[H");

    printf("\033[1;36m============================ MOTOR REAL-TIME MONITOR ============================\033[0m\n");
    printf("%-6s | %-10s | %-16s | %-22s | %-8s | %-8s | %-7s | %-16s | %-14s | %-14s | %-12s\n",
        "ID", "OFFLINE_CNT", "STEP", "MODE_SWITCH_STEP", "RX_MODE", "TX_MODE", "DES_EN",
        "TX_TARGET", "RX_POS_RAD", "RX_POS_DEG", "RX_VEL_RPM");
    printf("------------------------------------------------------------------------------------------------------------------------------------------------------\n");

    for (const auto& m : _motors) {
        if (std::find(print_motor_ids_.begin(), print_motor_ids_.end(), m.slave_index) == print_motor_ids_.end()) {
            continue;
        }
        const char* color_code = "\033[32m";
        if (m.step == MotorStep::FAULT) color_code = "\033[31m";
        if (m.step == MotorStep::STOPPED) color_code = "\033[35m";
        if (m.step == MotorStep::MODE_SWITCHING) color_code = "\033[33m";

        const char* step_name = "UNKNOWN";
        switch (m.step) {
            case MotorStep::IDLE: step_name = "IDLE"; break;
            case MotorStep::ENABLING: step_name = "ENABLING"; break;
            case MotorStep::RUNNING: step_name = "RUNNING"; break;
            case MotorStep::STOPPED: step_name = "STOPPED"; break;
            case MotorStep::FAULT: step_name = "FAULT"; break;
            case MotorStep::MODE_SWITCHING: step_name = "MODE_SWITCHING"; break;
        }

        const char* mode_switch_step_name = "N/A";
        switch (m.mode_switch_step) {
            case ModeSwitchStep::IDLE: mode_switch_step_name = "IDLE"; break;
            case ModeSwitchStep::SET_MODE: mode_switch_step_name = "SET_MODE"; break;
            case ModeSwitchStep::CLEAR: mode_switch_step_name = "CLEAR"; break;
            case ModeSwitchStep::DISABLE: mode_switch_step_name = "DISABLE"; break;
            case ModeSwitchStep::ENABLE: mode_switch_step_name = "ENABLE"; break;
            case ModeSwitchStep::OPERATING: mode_switch_step_name = "OPERATING"; break;
            case ModeSwitchStep::DONE: mode_switch_step_name = "DONE"; break;
        }

        const char* mode_name = "UNKNOWN";
        switch (m.rx.op_mode) {
            case ControlMode::NONE: mode_name = "NONE"; break;
            case ControlMode::PVT: mode_name = "PVT"; break;
            case ControlMode::CSP: mode_name = "CSP"; break;
            case ControlMode::CSV: mode_name = "CSV"; break;
            case ControlMode::CST: mode_name = "CST"; break;
            default: break;
        }

        const char* tx_mode_name = "UNKNOWN";
        switch (m.tx.op_mode) {
            case ControlMode::NONE: tx_mode_name = "NONE"; break;
            case ControlMode::PVT: tx_mode_name = "PVT"; break;
            case ControlMode::CSP: tx_mode_name = "CSP"; break;
            case ControlMode::CSV: tx_mode_name = "CSV"; break;
            case ControlMode::CST: tx_mode_name = "CST"; break;
            default: break;
        }

        const double rx_pos_rad = raw_pos_to_rad_for_motor(
            static_cast<double>(m.rx.pos), m.slave_index);
        const double rx_pos_deg = rx_pos_rad * kRadToDeg;
        const double rx_vel_rad_s = raw_vel_to_rad_s_for_motor(
            static_cast<double>(m.rx.vel), m.slave_index);
        const double rx_vel_rpm = rx_vel_rad_s * 60.0 / (2.0 * kPi);
        char tx_target_info[64] = {};
        switch (m.tx.op_mode) {
            case ControlMode::PVT:
                std::snprintf(tx_target_info, sizeof(tx_target_info), "p=%.6f kp=%.3f kd=%.3f",
                    myactua::motor_units::raw_pos_to_rad(
                        static_cast<double>(m.tx.target_pos), m.slave_index),
                    static_cast<double>(m.tx.pvt_kp) * 0.001,
                    static_cast<double>(m.tx.pvt_kd) * 0.001);
                break;
            case ControlMode::CSP:
                std::snprintf(tx_target_info, sizeof(tx_target_info), "%.6f rad",
                    myactua::motor_units::raw_pos_to_rad(
                        static_cast<double>(m.tx.target_pos), m.slave_index));
                break;
            case ControlMode::CSV:
                std::snprintf(tx_target_info, sizeof(tx_target_info), "%.3f rpm",
                    myactua::motor_units::raw_vel_to_rad_s(
                        static_cast<double>(m.tx.target_vel), m.slave_index) *
                    60.0 / (2.0 * kPi));
                break;
            case ControlMode::CST:
                std::snprintf(tx_target_info, sizeof(tx_target_info), "%d raw",
                    static_cast<int>(m.tx.target_torque));
                break;
            default:
                std::snprintf(tx_target_info, sizeof(tx_target_info), "N/A");
                break;
        }

        printf("M %-4d | %-10u | %s%-16s\033[0m | %s%-22s\033[0m | %-8s | %-8s | %-7s | %-16s | %-14.6f | %-14.3f | %-12.3f\n",
            m.slave_index,
            static_cast<unsigned int>(m.comm_offline_total_count),
            color_code,
            step_name,
            color_code,
            mode_switch_step_name,
            mode_name,
            tx_mode_name,
            m.desired.enabled ? "Y" : "N",
            tx_target_info,
            rx_pos_rad,
            rx_pos_deg,
            rx_vel_rpm);
    }
    printf("\033[1;36m=================================================================================\033[0m\n");
    
    fflush(stdout); 
}


bool MYACTUA::start()
{
    if (running_) return rt_start_permitted_.load(std::memory_order_acquire);

    running_ = true;
    rt_start_permitted_.store(false, std::memory_order_release);
    rt_thread_ = std::thread(&MYACTUA::rt_thread_func, this);

    sched_param requested_param{};
    requested_param.sched_priority = 80;
    const int sched_result = pthread_setschedparam(
        rt_thread_.native_handle(), SCHED_FIFO, &requested_param);

    int actual_policy = SCHED_OTHER;
    sched_param actual_param{};
    const int verify_sched_result = pthread_getschedparam(
        rt_thread_.native_handle(), &actual_policy, &actual_param);
    const bool sched_ready = sched_result == 0 && verify_sched_result == 0 &&
                             actual_policy == SCHED_FIFO &&
                             actual_param.sched_priority == requested_param.sched_priority;

    // 绑定到大核 cpu4，避免与 VS Code 等普通进程在 0-3 小核上竞争，
    // 也避免调度器把实时线程放到不可预测的核上导致周期抖动。
    cpu_set_t requested_cpuset;
    CPU_ZERO(&requested_cpuset);
    CPU_SET(4, &requested_cpuset);
    const int affinity_result = pthread_setaffinity_np(
        rt_thread_.native_handle(), sizeof(requested_cpuset), &requested_cpuset);

    cpu_set_t actual_cpuset;
    CPU_ZERO(&actual_cpuset);
    const int verify_affinity_result = pthread_getaffinity_np(
        rt_thread_.native_handle(), sizeof(actual_cpuset), &actual_cpuset);
    const bool affinity_ready = affinity_result == 0 && verify_affinity_result == 0 &&
                                CPU_ISSET(4, &actual_cpuset);

    if (!sched_ready || !affinity_ready) {
        std::cerr << "[MYACTUA] Failed to establish verified real-time control thread: "
                  << "setsched=" << sched_result
                  << ", getsched=" << verify_sched_result
                  << ", policy=" << actual_policy
                  << ", priority=" << actual_param.sched_priority
                  << ", setaffinity=" << affinity_result
                  << ", getaffinity=" << verify_affinity_result << std::endl;
        running_ = false;
        if (rt_thread_.joinable()) {
            rt_thread_.join();
        }
        return false;
    }

    rt_start_permitted_.store(true, std::memory_order_release);
    std::cout << "[MYACTUA] 实时控制线程已启动 (SCHED_FIFO prio=80, pinned cpu4)" << std::endl;
    return true;
}


void MYACTUA::shutdown()
{
    if (!running_) return;
    
    running_ = false;
    rt_start_permitted_.store(false, std::memory_order_release);
    if (rt_thread_.joinable()) {
        rt_thread_.join();
    }
    
    std::cout << "[MYACTUA] 实时控制线程已停止" << std::endl;
}


}
