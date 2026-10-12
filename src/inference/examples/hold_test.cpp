/* hold_test — 纯位置保持，验证训练站立姿态与并联脚踝 IK/FK 链路。 */
#include "robot_interface.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> g_stop{false};

void signal_handler(int)
{
    g_stop.store(true);
}

inference::RobotInterfaceConfig make_cfg()
{
    inference::RobotInterfaceConfig cfg;
    cfg.num_motors = 12;
    cfg.ethercat_ifname = "eth0";
    // 使用 MIT/PVT 阻抗控制，与训练配置一致。
    // 驱动侧的模式切换依赖 restart_motors() 补发的 SetMode 命令。
    cfg.motor_control_mode = myactua::ControlMode::PVT;

    // 非脚踝关节直接映射；两侧并联脚踝由 RobotInterface 内部 IK/FK 处理。
    cfg.model_to_motor_index = {0, 1, 2, 3, 6, 7, 8, 9};
    cfg.left_ankle_parallel = {4, 5, 4, 5};
    cfg.right_ankle_parallel = {10, 11, 10, 11};
    cfg.ankle_parallel_kinematics_enabled = true;

    cfg.motor_to_model_direction = {
        -1, -1, 1, 1, -1, -1,
        -1, 1, 1, -1, -1, -1,
    };

    // 训练约定的站立姿态（walking/standing 统一）：
    // hip_pitch=-0.3, knee_pitch=0.45, ankle_pitch=-0.15
    cfg.stand_pose_rad = {
        0.0, -0.3, 0.0, 0.45, -0.15, 0.0,
        0.0, -0.3, 0.0, 0.45, -0.15, 0.0,
    };
    cfg.joint_min_rad.assign(12, -1.0);
    cfg.joint_max_rad.assign(12, 1.5);

    // V2 站立策略相同的 PVT/MIT 刚度和阻尼。
    cfg.mit_kp = {220, 250, 220, 250, 20, 20, 220, 250, 220, 250, 20, 20};
    cfg.mit_kd = {8, 10, 8, 10, 1, 1, 8, 10, 8, 10, 1, 1};
    cfg.print_motors_info = false;
    return cfg;
}

void shutdown(inference::RobotInterface& robot)
{
    robot.stop_motors(-1);
    robot.deinit_motors();
}

}  // namespace

int main()
{
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    for (int remaining = 3; remaining > 0 && !g_stop.load(); --remaining) {
        std::cout << "[INFO] training pose hold starts in " << remaining
                  << " s. Press Ctrl+C to cancel.\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (g_stop.load()) return 0;

    const auto cfg = make_cfg();
    inference::RobotInterface robot(cfg);

    std::cout << "[1/3] Starting EtherCAT motors...\n";
    if (!robot.initial_and_start_motors()) {
        shutdown(robot);
        return 1;
    }

    std::cout << "[2/3] Enabling motors...\n";
    if (!robot.restart_motors(-1)) {
        shutdown(robot);
        return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // reset_joints 通过完整的模型映射与脚踝 IK 轨迹插值到训练站立姿态。
    std::cout << "[3/3] Resetting to training pose: hip=-0.3, knee=0.45, ankle=-0.15 rad...\n";
    if (!robot.reset_joints()) {
        shutdown(robot);
        return 1;
    }

    std::cout << "[INFO] Holding training pose. Ctrl+C to stop.\n";
    const auto period = std::chrono::milliseconds(20);
    auto next_tick = std::chrono::steady_clock::now();
    int tick = 0;
    while (!g_stop.load()) {
        next_tick += period;
        if (!robot.apply_action(cfg.stand_pose_rad)) {
            std::cerr << "[ERROR] Failed to hold training pose.\n";
            break;
        }
        // 每秒打印一次“目标 vs 实际”，数据即可判定电机有没有真正跟随。
        if (++tick % 50 == 0) {
            const std::vector<double> q = robot.get_joint_q();
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "[DIAG] 目标(模型序rad):";
            for (double v : cfg.stand_pose_rad) std::cout << ' ' << v;
            std::cout << "\n[DIAG] 实际(电机序rad):";
            for (double v : q) std::cout << ' ' << v;
            std::cout << '\n';
        }
        std::this_thread::sleep_until(next_tick);
    }

    shutdown(robot);
    std::cout << "[INFO] Done.\n";
    return 0;
}
