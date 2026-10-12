#include "robot_interface.hpp"
#include "EthercatAdapterIGH.hpp"
#include "motor_control.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <thread>
#include <vector>
#include <string>

namespace {
std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running.store(false);
}

const char* kJointName[12] = {
    "L_hip_roll",   // 0
    "L_hip_pitch",  // 1
    "L_hip_yaw",    // 2
    "L_knee",       // 3
    "L_ankle_up",   // 4
    "L_ankle_low",  // 5
    "R_hip_roll",   // 6
    "R_hip_pitch",  // 7
    "R_hip_yaw",    // 8
    "R_knee",       // 9
    "R_ankle_low",  // 10
    "R_ankle_up",   // 11
};
}  // namespace

int main()
{
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // ---- 配置，和 policy_test 一致的 PD 参数 ----
    inference::RobotInterfaceConfig cfg;
    cfg.num_motors = 12;
    cfg.model_to_motor_index = {0, 1, 2, 3, 4, 5,
                                6, 7, 8, 9, 11, 10};
    cfg.mit_kp = {220, 250, 220, 250, 20, 20,
                  220, 250, 220, 250, 20, 20};
    cfg.mit_kd = {8, 10, 8, 10, 1, 1,
                  8, 10, 8, 10, 1, 1};
    cfg.print_motors_info = false;
    cfg.ankle_parallel_kinematics_enabled = false;  // 只测电机本身，不解算脚踝

    // RobotInterface 控制电机
    inference::RobotInterface robot(cfg);

    std::cout << "[1/4] 初始化电机...\n";
    if (!robot.initial_and_start_motors()) {
        std::cerr << "[错误] 失败\n";
        return 1;
    }

    std::cout << "[2/4] 重启电机...\n";
    if (!robot.restart_motors(-1)) {
        std::cerr << "[错误] 失败\n";
        robot.deinit_motors();
        return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::cout << "[3/4] 发送零位指令 (电机带力保持 0°)\n";
    std::vector<double> zero_cmd(12, 0.0);
    if (!robot.apply_action(zero_cmd)) {
        std::cerr << "[错误] 零位指令失败\n";
        robot.deinit_motors();
        return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // 通过底层 MYACTUA 获取状态（RobotInterface 没有直接暴露 comm_ok）
    // 但 get_motor_comm_status() 是我新加的接口
    // 实际我们用 robot.get_motor_comm_status() + robot.get_joint_q()

    std::cout << "[4/4] 开始监控 (Ctrl+C 停止)\n\n";

    // 禁用内置 monitor 打印
    // 无法直接访问 controller，但我们可以通过 RobotInterface 的方法获取数据
    // 对于更详细的诊断，可以单独跑 ID_test

    uint32_t offline_count[12] = {0};
    bool prev_ok[12] = {false};

    auto init_comm = robot.get_motor_comm_status();
    for (int i = 0; i < 12 && i < static_cast<int>(init_comm.size()); ++i) {
        prev_ok[i] = init_comm[i];
    }

    // 脚踝误差记录（全过程最小误差）
    const int ankle_idx[4] = {4, 5, 10, 11};  // L_up, L_low, R_low, R_up
    double ankle_min_err_deg[4] = {999.0, 999.0, 999.0, 999.0};
    double ankle_max_err_deg[4] = {0.0, 0.0, 0.0, 0.0};

    std::printf("\033[2J\033[H");
    uint64_t frame = 0;

    while (g_running.load()) {
        // 持续重发零位指令（电机保持位置）
        if (!robot.apply_action(zero_cmd)) {
            std::cerr << "[错误] 重发指令失败\n";
            break;
        }

        auto q_rad      = robot.get_joint_q();
        auto dq_rad_s   = robot.get_joint_vel();
        auto comm_ok    = robot.get_motor_comm_status();

        // 更新离线计数
        for (int i = 0; i < 12 && i < static_cast<int>(comm_ok.size()); ++i) {
            if (prev_ok[i] && !comm_ok[i]) {
                offline_count[i]++;
            }
            prev_ok[i] = comm_ok[i];
        }

        // 记录脚踝全过程最小/最大误差
        for (int ai = 0; ai < 4; ++ai) {
            int i = ankle_idx[ai];
            double deg = (i < static_cast<int>(q_rad.size()))
                         ? q_rad[i] * 180.0 / 3.14159265 : 0.0;
            double err = std::abs(deg);  // target=0, so err = |actual|
            if (err < ankle_min_err_deg[ai]) ankle_min_err_deg[ai] = err;
            if (err > ankle_max_err_deg[ai]) ankle_max_err_deg[ai] = err;
        }

        // 绘制
        std::printf("\033[H");
        std::printf("P1 电机监控 (带力保持 0°)  frame=%llu  Ctrl+C 停止\n",
                    static_cast<unsigned long long>(frame));
        std::printf("=================================================================="
                    "=====================\n");
        std::printf("M  | 关节名         | Target ° | Actual ° | Error ° | Comm |"
                    " Offline\n");
        std::printf("------------------------------------------------------------------"
                    "---------------------\n");

        for (int i = 0; i < 12; ++i) {
            double target_deg = 0.0;
            double actual_deg = (i < static_cast<int>(q_rad.size()))
                                ? q_rad[i] * 180.0 / 3.14159265 : 0.0;
            double err_deg = target_deg - actual_deg;
            bool ok = (i < static_cast<int>(comm_ok.size())) ? comm_ok[i] : false;

            const char* comm_str = ok ? "\033[32m✓\033[0m" : "\033[31m✗\033[0m";
            const char* offline_mark = (offline_count[i] > 0) ? "\033[33m★\033[0m" : " ";

            std::printf("M%-2d | %-14s | %+8.3f | %+8.3f | %+8.3f | %s  | %5u %s\n",
                        i, kJointName[i],
                        target_deg, actual_deg, err_deg,
                        comm_str, offline_count[i], offline_mark);
        }

        std::printf("=================================================================="
                    "=====================\n");
        std::printf("抱起机器人后看脚踝能否归零 (min err)    ");
        for (int ai = 0; ai < 4; ++ai) {
            std::printf("M%d: %.1f°  ", ankle_idx[ai], ankle_min_err_deg[ai]);
        }
        std::printf("\n");
        std::fflush(stdout);

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        ++frame;
    }

    robot.stop_motors(-1);
    robot.deinit_motors();

    // 最终统计
    int total_offline = 0;
    for (int i = 0; i < 12; ++i) {
        if (offline_count[i] > 0) {
            if (total_offline == 0) std::printf("\n");
            std::printf("M%d (%s): %u 次掉线\n", i, kJointName[i], offline_count[i]);
            total_offline += offline_count[i];
        }
    }
    if (total_offline == 0) {
        std::printf("\n无掉线事件 ✓\n");
    }

    // 脚踝最小误差总结
    std::printf("\n====== 脚踝归零测试总结 ======\n");
    std::printf("全程记录的最小误差（越接近0°越好）:\n");
    for (int ai = 0; ai < 4; ++ai) {
        int i = ankle_idx[ai];
        std::printf("  M%d (%s): min_err=%.2f°   max_err=%.2f°\n",
                    i, kJointName[i], ankle_min_err_deg[ai], ankle_max_err_deg[ai]);
    }
    bool all_ok = true;
    for (int ai = 0; ai < 4; ++ai) {
        if (ankle_min_err_deg[ai] > 2.0) { all_ok = false; break; }
    }
    if (all_ok) {
        std::printf("结论: ✅ 四个脚踝最小误差均 < 2°，归零正常！\n");
    } else {
        std::printf("结论: ❌ 有脚踝最小误差 > 2°，可能未完全归零\n");
    }
    std::printf("==============================\n");

    return 0;
}
