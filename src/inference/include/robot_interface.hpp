#pragma once
#include "EthercatAdapterIGH.hpp"
#include "imu_reader.hpp"
#include "motor_control.hpp"
#include "ankle_motor_ik.hpp"
#include "ankle_motor_fk.hpp"
#include "ankle_motor_jacobian.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace myactua {
class EthercatAdapterIGH;
class MYACTUA;
}

namespace imu {
class IMUReader;
}

namespace inference {

class TorchPolicyRunner;

/* 接口层的配置结构体 */
struct RobotInterfaceConfig {
    struct AnkleParallelMap {
        int model_pitch_dof = -1;
        int model_roll_dof = -1;
        int upper_motor_index = -1;
        int lower_motor_index = -1;
    };

    /* 电机 配置*/
    int num_motors = 12;
    std::string ethercat_ifname = "enp8s0";
    /* 等待所有从站就绪的超时和轮询时间 */
    int wait_all_slaves_timeout_ms = 30000;
    int wait_all_slaves_poll_ms    = 100;

    /* 是否在终端打印电机信息 */
    bool print_motors_info = false;  // false将 print_motor_ids 清零
    std::vector<int> print_motor_ids = {-1};  // 仅打印这些ID的电机信息，ID从0开始，-1表示全部

    /* 相对 stand_pose_rad 的关节弧度偏移限制，按模型 DOF 序号填写 */
    std::vector<double> joint_min_rad;
    std::vector<double> joint_max_rad;
    /* 初始姿态关节弧度值，按模型 DOF 序号填写 */
    std::vector<double> stand_pose_rad = {};
    /* 默认使用 MIT/PVT 模式；若改为 CSP，apply_action 将沿用位置指令路径 */
    myactua::ControlMode motor_control_mode = myactua::ControlMode::PVT;
    /* MIT/PVT 模式刚度/阻尼，必须由调用侧显式配置 */
    std::vector<double> mit_kp;
    std::vector<double> mit_kd;

    /* IMU 配置 */
    std::string imu_device = "/dev/ttyUSB0";
    int imu_baudrate       = 921600;

    bool imu_print_imu     = false;
    bool imu_print_ahrs    = true;
    bool imu_print_stats   = false;


    /* TorchScript 策略配置 */
    /* .pt 模型文件路径，由 torch.jit.script/trace 导出 */
    std::string policy_model_path;
    /* 模型关节维度到电机逻辑索引的映射，长度必须为 12 且不可重复；并联脚踝驱动链由 AnkleParallelMap 指定 */
    std::vector<int> model_to_motor_index;
    /* 电机物理/控制器 ID 到模型关节方向的符号；1 表示方向一致，-1 表示方向相反；为空时全部按 1 */
    std::vector<int> motor_to_model_direction;
    /* 缩放后动作偏移的逐 DOF 截断范围：{lower, upper}；按模型 DOF 序号使用 */
    std::vector<std::array<double, 2>> action_clip;
    /* 模型原始输出动作的缩放系数，长度必须为 12；按模型 DOF 序号使用 */
    std::vector<double> action_scale;
    /* 步态相位周期，单位 s；phase = fmod(elapsed / policy_cycle_time_s, 1) */
    double policy_cycle_time_s = 0.02;
    /* 站立策略使用训练时固定的 phase=(sin=0, cos=1)。 */
    bool fixed_policy_phase = false;
    /* DOF Pos 缩放，长度必须为 12；按模型 DOF 序号使用 */
    std::vector<double> dof_pos_scale;
    /* DOF Vel 缩放，长度必须为 12；按模型 DOF 序号使用 */
    std::vector<double> dof_vel_scale;
    /* 速度指令缩放，对应 vx、vy、yaw_rate */
    std::array<double, 3> command_scale = {
        0.1,
        0.1,
        0.1
    };
    /* 机身角速度缩放，对应 AHRS roll_speed、pitch_speed、heading_speed */
    std::array<double, 3> body_ang_vel_scale = {
        0.2,
        0.2,
        0.2
    };
    /* 机身欧拉角缩放，对应 AHRS roll、pitch、heading */
    std::array<double, 3> euler_scale = {
        1,
        1,
        1
    };

    /* 是否启用脚踝并联机构运动学；启用后 action 走 IK，observation 走 FK */
    bool ankle_parallel_kinematics_enabled = false;
    /* 并联脚踝显式映射：模型姿态 DOF 与两条物理驱动链电机分开配置 */
    AnkleParallelMap left_ankle_parallel;
    AnkleParallelMap right_ankle_parallel;
    /* 可选的物理踝电机角度限位，顺序为左 upper/lower、右 upper/lower */
    std::array<double, 4> ankle_motor_min_rad = {
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()
    };
    std::array<double, 4> ankle_motor_max_rad = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()
    };
};


class RobotInterface {
public:
    /* 构造函数中加载 RobotInterfaceConfig 到 config_ 进行参数配置 */
    explicit RobotInterface(RobotInterfaceConfig config = {});
    ~RobotInterface();

    /* 仅当电机与 IMU 都已初始化完成时返回 true */
    bool is_initialized() const {
        return motors_initialized_.load() && imu_initialized_.load();
    }


    /* 电机部分 */
    bool initial_and_start_motors();
    bool stop_motors(int slave_index = -1);
    bool restart_motors(int slave_index = -1);
    void deinit_motors();
    bool apply_action(const std::vector<double>& target_q_model_rad);  // 模型 DOF 顺序目标角(rad)
    bool send_motor_rad(const std::vector<double>& target_rad);  // 直接发电机目标(rad)，跳过模型映射
    bool reset_joints();  // /* 复位到模型 DOF 顺序配置的 stand_pose_rad，单位为 rad */

    bool is_motors_initialized() const { return motors_initialized_.load(); }
    std::vector<double> get_joint_q() const;    // rad
    std::vector<double> get_joint_vel() const;  // rad/s
    std::vector<double> get_joint_tau() const;  // raw driver unit
    std::vector<bool> get_motor_comm_status() const;  // 每个电机的通信状态
    void calibrate_encoder_zero();  // 标定当前编码器位置为零位
    void set_motor_target_offset(const std::vector<double>& offset);  // 设置电机目标偏移（model_dof顺序）


    /* IMU部分 */
    bool initial_and_start_imu();
    void deinit_imu();
    bool is_imu_initialized() const { return imu_initialized_.load(); }
    std::array<double, 4> get_quat() const;     // [w, x, y, z]
    std::array<double, 3> get_ang_vel() const;  // [wx, wy, wz], rad/s
    std::array<double, 3> get_body_ang_vel() const;  // [roll_speed, pitch_speed, heading_speed], rad/s
    std::array<double, 3> get_euler() const;          // [roll, pitch, heading], rad
    std::array<double, 3> get_projected_gravity() const;  // [x, y, z], unit vector in body frame



    /* 模型部分 */
    bool load_policy();
    void unload_policy();
    void reset_policy_state();
    void set_target_velocity(double vx, double vy, double yaw_rate);
    std::array<double, 3> get_target_velocity() const;
    bool policy_step();
    bool policy_step(double vx, double vy, double yaw_rate);


private:
    static constexpr int kPolicyDof = 12;  // 模型输出的动作维度，必须与电机数量一致
#if POLICY_V3
    static constexpr int kPolicySingleObservationSize = 47;  // 单帧模型观测维度
    static constexpr int kPolicyFrameStack = 15;             // 模型输入使用 15 帧观测
#else
    static constexpr int kPolicySingleObservationSize = 45;  // 单周期各 observation term 总维度
    static constexpr int kPolicyFrameStack = 5;              // policy.pt 每个 term 使用 5 帧历史
#endif
    static constexpr int kPolicyObservationSize = kPolicySingleObservationSize * kPolicyFrameStack;

    /* 机器人接口配置 */
    RobotInterfaceConfig config_;

    /* 以太网适配器 智能指针 */
    std::shared_ptr<myactua::EthercatAdapterIGH> adapter_;
    std::unique_ptr<myactua::MYACTUA> controller_;  // 电机控制器指针
    bool validate_motor_config() const;


    std::atomic<bool> motors_initialized_{false};
    // Gate apply_action: startup policy requires explicit restart after initial STOP.
    std::atomic<bool> motion_enabled_{false};


    /* IMU 读取器 智能指针 */
    std::unique_ptr<imu::IMUReader> imu_reader_;
    std::atomic<bool> imu_initialized_{false};
    std::atomic<bool> ahrs_ready_{false};

    mutable std::mutex imu_mutex_;
    std::array<double, 4> quat_{1.0, 0.0, 0.0, 0.0};
    std::array<double, 3> ang_vel_{0.0, 0.0, 0.0};
    std::array<double, 3> body_ang_vel_{0.0, 0.0, 0.0};
    std::array<double, 3> euler_{0.0, 0.0, 0.0};
    std::array<double, 3> projected_gravity_{0.0, 0.0, -1.0};
    bool projected_gravity_valid_ = false;


    /*  Policy 运行器 智能指针 */
    std::unique_ptr<TorchPolicyRunner> policy_runner_;

    // 编码器零偏（calibrate_encoder_zero 后生效）
    std::vector<double> encoder_home_offset_;
    bool encoder_calibrated_ = false;

    // 电机目标偏移（用于 stand_test 过渡到直立）
    std::vector<double> motor_target_offset_;

    mutable std::mutex policy_mutex_;
    mutable std::mutex target_velocity_mutex_;
    std::array<double, 3> target_velocity_{0.0, 0.0, 0.0};  // [vx, vy, yaw_rate]
    std::array<float, kPolicyDof> last_action_raw_{};  // 记录上一周期动作值
    std::array<float, kPolicyObservationSize> observation_history_{};
    bool observation_history_ready_ = false;
    std::chrono::steady_clock::time_point policy_start_time_{};
    std::ofstream policy_observation_log_;
    std::uint64_t policy_observation_frame_index_ = 0;
    bool policy_observation_log_failed_ = false;
    std::ofstream policy_output_log_;
    std::uint64_t policy_output_frame_index_ = 0;
    bool policy_output_log_failed_ = false;

    bool validate_policy_config() const;
    bool build_action_target_rad(const std::vector<double>& target_q_model_rad,
                                 std::vector<double>& target_rad) const;
    bool is_parallel_ankle_model_dof(int model_index) const;
    int direct_model_mapping_slot(int model_index) const;
    bool build_policy_observation(double vx,
                                  double vy,
                                  double yaw_rate,
                                  std::array<float, kPolicyObservationSize>& observation);
    bool build_joint_observation_terms(const std::vector<double>& q_rad,
                                       const std::vector<double>& dq_rad_s,
                                       std::array<float, kPolicyDof>& joint_pos_rel,
                                       std::array<float, kPolicyDof>& joint_vel_rel);
    bool apply_ankle_fk_observation(
        const std::vector<double>& q_rad,
        const std::vector<double>& dq_rad_s,
        const RobotInterfaceConfig::AnkleParallelMap& ankle_map,
        ankle_motor_fk::Solver& solver,
        std::array<float, kPolicyDof>& joint_pos_rel,
        std::array<float, kPolicyDof>& joint_vel_rel);
    void reset_ankle_fk_state();
    void reset_policy_observation_log();
    bool ensure_policy_observation_log();
    void log_policy_observation_frame(
        const std::array<float, kPolicyObservationSize>& observation);
    void reset_policy_output_log();
    bool ensure_policy_output_log();
    void log_policy_output(const std::array<float, kPolicyDof>& raw_action,
                           const std::vector<double>& target_q_model_rad,
                           const std::vector<double>& target_rad);
    bool handle_policy_step_failure(const std::string& message);

    /* 脚踝 IK 求解器（左右各一），状态跨 policy_step 保持 */
    mutable ankle_motor_ik::Solver left_ankle_solver_;
    mutable ankle_motor_ik::Solver right_ankle_solver_;
    /* 上一次成功求解的驱动链电机角度（弧度），用于不可达时回退 */
    mutable double left_ankle_last_upper_motor_ = 0.0;
    mutable double left_ankle_last_lower_motor_ = 0.0;
    mutable double right_ankle_last_upper_motor_ = 0.0;
    mutable double right_ankle_last_lower_motor_ = 0.0;
    mutable bool left_ankle_solved_ = false;
    mutable bool right_ankle_solved_ = false;

    /* 脚踝 FK 求解器：真实电机反馈 -> 脚踝 roll/pitch observation */
    ankle_motor_fk::Solver left_ankle_fk_solver_;
    ankle_motor_fk::Solver right_ankle_fk_solver_;

    /* 对单条腿的脚踝并联机构执行逆解，将 roll/pitch 转为两个电机角度 */
    bool apply_ankle_ik(const std::vector<double>& target_q_model_rad,
                        std::vector<double>& target_rad,
                        const RobotInterfaceConfig::AnkleParallelMap& ankle_map,
                        ankle_motor_ik::Solver& solver,
                        double& last_upper_motor, double& last_lower_motor,
                        bool& solved) const;
};

}  // namespace inference
