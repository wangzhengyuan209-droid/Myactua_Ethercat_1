#include "robot_interface.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr const char* kPolicyModelPath = "../model/policy_v3.pt";

constexpr const char* kEthercatIfname = "eth0";
constexpr const char* kImuDevice = "/dev/ttyUSB0";
constexpr int kImuBaudrate = 921600;

constexpr double kPolicyHz = 50.0;
constexpr double kPolicyPeriodSec = 1.0 / kPolicyHz;
constexpr double kCommandVx = 0.0;
constexpr double kCommandVy = 0.0;
constexpr double kCommandYawRate = 0.0;

std::atomic<bool> g_stop_requested{false};

void signal_handler(int)
{
    g_stop_requested.store(true);
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
    cfg.imu_device      = kImuDevice;
    cfg.imu_baudrate    = kImuBaudrate;
    cfg.imu_print_imu   = false;
    cfg.imu_print_ahrs  = true;
    cfg.imu_print_stats = false;

    cfg.policy_model_path = kPolicyModelPath;

    cfg.print_motors_info = true;
    cfg.print_motor_ids = {0, 1, 2, 3, 4, 5,
                           6, 7, 8, 9, 10, 11};

#if POLICY_V3
    // 并联脚踝由 IK/FK 处理，因此这里只保留 8 个普通关节的物理电机映射。
    cfg.model_to_motor_index = {0, 1, 2, 3, 6, 7, 8, 9};
    cfg.left_ankle_parallel = {4, 5, 4, 5};      // L: pitch, roll, upper(M4), lower(M5)
    cfg.right_ankle_parallel = {10, 11, 10, 11}; // R: pitch, roll, upper(M10), lower(M11)
    // 参考学长配置，按物理电机顺序限制踝驱动链目标角度(rad)。
    cfg.ankle_motor_min_rad = {-1.1, -1.1, -1.1, -1.1};
    cfg.ankle_motor_max_rad = { 1.1,  1.1,  1.1,  1.1};
#else
    cfg.model_to_motor_index = {0, 1, 2, 3, 4, 5,
                                6, 7, 8, 9, 10, 11};
#endif

    // 与训练一致：action_scale=0.25，部署安全限幅仍限制最终偏移。
    cfg.action_clip.assign(12, {-0.10, 0.10});

    // 训练配置的 action_scale=0.25；保留部署侧保守动作限幅，避免首次上机出现大幅动作。
    cfg.action_scale.assign(12, 0.25);

    cfg.joint_min_rad.assign(12, -1);
    cfg.joint_max_rad.assign(12,  1.5);

    cfg.policy_cycle_time_s = 0.4;  // gait cycle time, 和训练一致

    // 与 walking 训练一致的默认姿态：hip_pitch=-0.3、knee_pitch=0.45、ankle_pitch=-0.15。
    cfg.stand_pose_rad = {
        0.0, -0.3, 0.0, 0.45, -0.15, 0.0,
        0.0, -0.3, 0.0, 0.45, -0.15, 0.0
    };

    cfg.dof_pos_scale.assign(12, 1.0);
    cfg.dof_vel_scale.assign(12, 0.05);

    cfg.motor_to_model_direction = {
        -1, -1,  1,  1,  -1, -1,   // M0-M5: L_hip(roll,pitch,yaw,knee,ankle_up,ankle_low)
        -1,  1,  1, -1,  -1, -1    // M6-M11: R_hip(roll,pitch,yaw,knee,ankle_low,ankle_up)
    };

    /******************************************************************* */

    cfg.command_scale = {2.0, 2.0, 1.0};
    cfg.body_ang_vel_scale = {1.0, 1.0, 1.0};
    cfg.euler_scale = {1.0, 1.0, 1.0};

    cfg.mit_kp = {220, 250, 220, 250, 20, 20, 220, 250, 220, 250, 20, 20};
    cfg.mit_kd = {8, 10, 8, 10, 1, 1, 8, 10, 8, 10, 1, 1};

    // ===== 并联脚踝配置（学长的 AnkleParallelMap 方式）=====
    cfg.ankle_parallel_kinematics_enabled = true;

    return cfg;
}

bool safety_countdown()
{
    for (int remaining = 3; remaining > 0; --remaining) {
        if (g_stop_requested.load()) {
            return false;
        }
        std::cout << "[INFO] Starting hardware in " << remaining
                  << " seconds. Press Ctrl+C to cancel.\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return !g_stop_requested.load();
}

void safe_shutdown(inference::RobotInterface& robot)
{
    std::cout << "[INFO] Stopping motors and releasing hardware...\n";
    robot.stop_motors(-1);
    robot.deinit_motors();
    robot.deinit_imu();
    std::cout << "[INFO] Shutdown complete.\n";
}

}  // namespace

int main()
{
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    const inference::RobotInterfaceConfig cfg = make_robot_config();

    if (!file_readable(cfg.policy_model_path)) {
        std::cerr << "[ERROR] Policy model is not readable: "
                  << cfg.policy_model_path << "\n"
                  << "[ERROR] Put the TorchScript model at "
                  << "src/inference/model/policy.pt and run from "
                  << "src/inference/build.\n";
        return 1;
    }

    if (!safety_countdown()) {
        std::cout << "[INFO] Startup canceled before hardware initialization.\n";
        return 0;
    }

    inference::RobotInterface robot(cfg);

    // 学长推荐的启动顺序：EtherCAT → load policy → restart(enable) → IMU → reset → 等站稳 → policy
    std::cout << "[INFO] Starting EtherCAT motors...\n";
    if (!robot.initial_and_start_motors()) {
        std::cerr << "[ERROR] initial_and_start_motors() failed.\n";
        safe_shutdown(robot);
        return 1;
    }

    std::cout << "[INFO] Loading policy...\n";
    if (!robot.load_policy()) {
        std::cerr << "[ERROR] load_policy() failed.\n";
        safe_shutdown(robot);
        return 1;
    }

    std::cout << "[INFO] Restarting motors...\n";
    if (!robot.restart_motors(-1)) {
        std::cerr << "[ERROR] restart_motors(-1) failed.\n";
        safe_shutdown(robot);
        return 1;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    if (g_stop_requested.load()) {
        safe_shutdown(robot);
        return 0;
    }

    std::cout << "[INFO] Starting IMU...\n";
    if (!robot.initial_and_start_imu()) {
        std::cerr << "[ERROR] initial_and_start_imu() failed.\n";
        safe_shutdown(robot);
        return 1;
    }

    std::cout << "[INFO] Waiting 1 second for IMU/AHRS warmup...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (g_stop_requested.load()) {
        safe_shutdown(robot);
        return 0;
    }

    std::cout << "[INFO] Resetting joints to stand_pose_rad...\n";
    if (!robot.reset_joints()) {
        std::cerr << "[ERROR] reset_joints() failed.\n";
        safe_shutdown(robot);
        return 1;
    }

    // 等待机器人站稳，之后进入策略循环
    std::cout << "[INFO] Waiting 5s for stabilization, then 2s policy recording...\n";
    std::this_thread::sleep_for(std::chrono::seconds(5));

    std::cout << "[INFO] Entering policy loop for 2 seconds. Will auto-exit.\n";
    using Clock = std::chrono::steady_clock;
    const auto period = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(kPolicyPeriodSec));
    auto next_tick = Clock::now();
    auto last_report = Clock::now();

    std::uint64_t steps = 0;
    double total_step_ms = 0.0;
    double max_step_ms = 0.0;

    constexpr std::uint64_t kMaxSteps = 100;  // ~2 seconds at 50Hz
    while (!g_stop_requested.load() && steps < kMaxSteps) {
        next_tick += period;

        const auto step_start = Clock::now();
        if (!robot.policy_step(kCommandVx, kCommandVy, kCommandYawRate)) {
            std::cerr << "[ERROR] policy_step() failed at step " << steps << ".\n";
            safe_shutdown(robot);
            return 1;
        }
        const auto step_end = Clock::now();

        const double step_ms =
            std::chrono::duration<double, std::milli>(step_end - step_start).count();
        total_step_ms += step_ms;
        max_step_ms = std::max(max_step_ms, step_ms);
        ++steps;

        const auto now = Clock::now();
        if (now - last_report >= std::chrono::seconds(1)) {
            const double avg_step_ms = total_step_ms / static_cast<double>(steps);
            std::cout << std::fixed << std::setprecision(3)
                      << "[INFO] steps=" << steps
                      << " avg_policy_step_ms=" << avg_step_ms
                      << " max_policy_step_ms=" << max_step_ms << "\n";
            last_report = now;
        }

        std::this_thread::sleep_until(next_tick);
    }

    const double avg_step_ms = steps > 0
        ? total_step_ms / static_cast<double>(steps)
        : 0.0;
    std::cout << std::fixed << std::setprecision(3)
              << "[INFO] Exiting policy loop. total_steps=" << steps
              << " avg_policy_step_ms=" << avg_step_ms
              << " max_policy_step_ms=" << max_step_ms << "\n";

    safe_shutdown(robot);
    return 0;
}
