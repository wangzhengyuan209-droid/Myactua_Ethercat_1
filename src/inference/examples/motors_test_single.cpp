#include "robot_interface.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>
#include <vector>

namespace {
std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running.store(false);
}
}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "用法: ./motors_test_single <电机编号>" << std::endl;
        std::cout << "例如: ./motors_test_single 0  (测试电机0)" << std::endl;
        std::cout << "电机编号范围: 0-11" << std::endl;
        return -1;
    }

    int motor_id = std::atoi(argv[1]);
    if (motor_id < 0 || motor_id > 11) {
        std::cerr << "错误: 电机编号必须在 0-11 之间" << std::endl;
        return -1;
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    inference::RobotInterfaceConfig cfg;

    // 设置 MIT 模式 PD 参数
    cfg.mit_kp.assign(12, 250);
    cfg.mit_kd.assign(12, 10);

    inference::RobotInterface robot(cfg);

    std::cout << "[1/4] 初始化电机..." << std::endl;
    if (!robot.initial_and_start_motors()) {
        std::cerr << "[错误] initial_and_start_motors failed." << std::endl;
        return -1;
    }

    std::cout << "[2/4] 重启电机..." << std::endl;
    if (!robot.restart_motors(-1)) {
        std::cerr << "[错误] restart_motors failed." << std::endl;
        robot.deinit_motors();
        return -1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "[3/4] 测试电机 " << motor_id << "，按 Ctrl+C 停止" << std::endl;
    std::cout << "电机 " << motor_id << " 在 -5 到 +5 度之间慢速摆动，其他电机保持 0 位" << std::endl;
    std::cout << "观察哪个关节在动..." << std::endl;

    // 5 度 = 0.0873 弧度
    constexpr double kMaxAngleRad = 5.0 * M_PI / 180.0;
    constexpr int kSteps = 50;           // 50 步从 -5 度到 +5 度
    constexpr int kStepDelayMs = 100;    // 每步 100ms，一个来回约 10 秒

    std::vector<double> action(12, 0.0);

    while (g_running.load()) {
        // 从 -5 度到 +5 度
        for (int i = 0; i <= kSteps && g_running.load(); ++i) {
            double angle = -kMaxAngleRad + 2.0 * kMaxAngleRad * i / kSteps;
            action[motor_id] = angle;
            robot.apply_action(action);
            std::this_thread::sleep_for(std::chrono::milliseconds(kStepDelayMs));
        }
        // 从 +5 度到 -5 度
        for (int i = kSteps; i >= 0 && g_running.load(); --i) {
            double angle = -kMaxAngleRad + 2.0 * kMaxAngleRad * i / kSteps;
            action[motor_id] = angle;
            robot.apply_action(action);
            std::this_thread::sleep_for(std::chrono::milliseconds(kStepDelayMs));
        }
    }

    // 回零
    robot.apply_action(std::vector<double>(12, 0.0));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    robot.deinit_motors();
    std::cout << "\n[完成] 测试结束。电机 " << motor_id << " 对应的关节是？" << std::endl;
    return 0;
}
