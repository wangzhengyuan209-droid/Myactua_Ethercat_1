#include "robot_interface.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <thread>

namespace {

constexpr const char* kPolicyModelPath = "../model/standing_policy_V2.pt";
constexpr const char* kEthercatIfname = "eth0";
constexpr const char* kImuDevice = "/dev/ttyUSB0";
constexpr int kImuBaudrate = 921600;
constexpr double kPolicyHz = 100.0;
constexpr auto kStabilizeDuration = std::chrono::seconds(5);

std::atomic<bool> g_stop_requested{false};

void signal_handler(int)
{
    g_stop_requested.store(true);
}

bool wait_cancellable(std::chrono::milliseconds duration)
{
    constexpr auto kPollPeriod = std::chrono::milliseconds(50);
    while (duration.count() > 0 && !g_stop_requested.load()) {
        const auto slice = std::min(duration, kPollPeriod);
        std::this_thread::sleep_for(slice);
        duration -= slice;
    }
    return !g_stop_requested.load();
}

bool file_readable(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

inference::RobotInterfaceConfig make_robot_config()
{
    inference::RobotInterfaceConfig cfg;
    cfg.num_motors = 12;
    cfg.ethercat_ifname = kEthercatIfname;
    // 显式指定 MIT/PVT 阻抗控制（与训练一致）。
    // 驱动侧的模式切换依赖 restart_motors() 补发的 SetMode 命令。
    cfg.motor_control_mode = myactua::ControlMode::PVT;
    cfg.imu_device = kImuDevice;
    cfg.imu_baudrate = kImuBaudrate;
    cfg.imu_print_imu = false;
    cfg.imu_print_ahrs = false;
    cfg.imu_print_stats = false;
    cfg.policy_model_path = kPolicyModelPath;

    // 非并联踝关节按模型顺序直接映射；两侧脚踝由 IK/FK 接管。
    cfg.model_to_motor_index = {0, 1, 2, 3, 6, 7, 8, 9};
    cfg.left_ankle_parallel = {4, 5, 4, 5};
    cfg.right_ankle_parallel = {10, 11, 10, 11}; // R: pitch, roll, upper(M10), lower(M11)
    // 参考学长配置，按物理电机顺序限制踝驱动链目标角度(rad)。
    cfg.ankle_motor_min_rad = {-1.1, -1.1, -1.1, -1.1};
    cfg.ankle_motor_max_rad = { 1.1,  1.1,  1.1,  1.1};
    cfg.ankle_parallel_kinematics_enabled = true;

    // 安全钳位：action_clip 直接裁剪缩放后偏移。已从 ±0.05 放宽到 ±0.10 rad(≈5.7°)，
    // 给策略更大的推搡恢复权限（训练侧无此限幅，仅 raw action ±18）。
    // 站立稳态动作本就接近 0，此上限不影响正常平衡，仅保护策略异常发散时机器人不大幅摆动。
    cfg.action_clip.assign(12, {-0.10, 0.10});
    cfg.action_scale.assign(12, 0.25);
    cfg.joint_min_rad.assign(12, -1.0);
    cfg.joint_max_rad.assign(12, 1.5);

    // 与 standing 训练配置一致：hip_pitch=-0.3、knee_pitch=0.45、ankle_pitch=-0.15。
    cfg.stand_pose_rad = {
        0.0, -0.3, 0.0, 0.45, -0.15, 0.0,
        0.0, -0.3, 0.0, 0.45, -0.15, 0.0,
    };
    cfg.dof_pos_scale.assign(12, 1.0);
    cfg.dof_vel_scale.assign(12, 0.05);
    cfg.command_scale = {2.0, 2.0, 1.0};
    cfg.body_ang_vel_scale = {1.0, 1.0, 1.0};
    cfg.euler_scale = {1.0, 1.0, 1.0};
    cfg.fixed_policy_phase = true;

    cfg.motor_to_model_direction = {
        -1, -1, 1, 1, -1, -1,
        -1, 1, 1, -1, -1, -1,
    };
    cfg.mit_kp = {220, 250, 220, 250, 20, 20, 220, 250, 220, 250, 20, 20};
    cfg.mit_kd = {8, 10, 8, 10, 1, 1, 8, 10, 8, 10, 1, 1};
    return cfg;
}

void safe_shutdown(inference::RobotInterface& robot)
{
    robot.stop_motors(-1);
    robot.deinit_motors();
    robot.deinit_imu();
}

}  // namespace

int main()
{
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // 策略线程绑定到独立大核 cpu5，与 EtherCAT 实时线程(cpu4) 分核，避免 CPU 竞争。
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(5, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        std::cout << "[INFO] policy thread pinned to cpu5\n";
    }

    const auto cfg = make_robot_config();
    if (!file_readable(cfg.policy_model_path)) {
        std::cerr << "[ERROR] Missing standing TorchScript model: "
                  << cfg.policy_model_path << "\n";
        return 1;
    }

    for (int remaining = 3; remaining > 0; --remaining) {
        std::cout << "[INFO] Starting standing policy in " << remaining
                  << " seconds. Press Ctrl+C to cancel.\n";
        if (!wait_cancellable(std::chrono::seconds(1))) return 0;
    }

    inference::RobotInterface robot(cfg);
    std::cout << "[INFO] Starting EtherCAT motors...\n";
    if (!robot.initial_and_start_motors()) {
        safe_shutdown(robot);
        return 1;
    }
    std::cout << "[INFO] Loading fixed-phase standing policy...\n";
    if (!robot.load_policy()) {
        safe_shutdown(robot);
        return 1;
    }
    if (!robot.restart_motors(-1) || !wait_cancellable(std::chrono::milliseconds(500))) {
        safe_shutdown(robot);
        return g_stop_requested.load() ? 0 : 1;
    }
    std::cout << "[INFO] Starting IMU...\n";
    if (!robot.initial_and_start_imu() || !wait_cancellable(std::chrono::seconds(1))) {
        safe_shutdown(robot);
        return g_stop_requested.load() ? 0 : 1;
    }
    std::cout << "[INFO] Resetting to standing pose...\n";
    if (!robot.reset_joints()) {
        safe_shutdown(robot);
        return 1;
    }
    std::cout << "[INFO] Stabilizing for " << kStabilizeDuration.count() << " s...\n";
    if (!wait_cancellable(std::chrono::duration_cast<std::chrono::milliseconds>(kStabilizeDuration))) {
        safe_shutdown(robot);
        return 0;
    }

    const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / kPolicyHz));
    auto next_tick = std::chrono::steady_clock::now();
    std::uint64_t steps = 0;
    std::cout << "[INFO] Standing policy running until Ctrl+C.\n";
    while (!g_stop_requested.load()) {
        next_tick += period;
        if (!robot.policy_step(0.0, 0.0, 0.0)) {
            std::cerr << "[ERROR] standing policy_step failed at step " << steps << "\n";
            safe_shutdown(robot);
            return 1;
        }
        ++steps;
        std::this_thread::sleep_until(next_tick);
        if (std::chrono::steady_clock::now() > next_tick + period) {
            next_tick = std::chrono::steady_clock::now();
        }
        if (steps % 250 == 0) {
            std::cout << "[INFO] standing steps=" << steps << "\n";
        }
    }

    std::cout << "[INFO] Standing test finished after " << steps << " steps.\n";
    safe_shutdown(robot);
    return 0;
}
