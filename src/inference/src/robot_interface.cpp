#include "robot_interface.hpp"
#include "torch_policy_runner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>


namespace inference {
namespace {

constexpr double kPi = 3.14159265358979323846;

/* build_policy_observation 每次成功构建的单帧观测写到这里。 */
const std::filesystem::path& policy_observation_log_dir()
{
    static const std::filesystem::path path =
        std::filesystem::path(__FILE__).parent_path().parent_path() /
        "log" / "policy_observation";
    return path;
}

/* policy_step 推理输出和 apply_action 映射后的电机目标写到这里。 */
const std::filesystem::path& policy_output_log_dir()
{
    static const std::filesystem::path path =
        std::filesystem::path(__FILE__).parent_path().parent_path() /
        "log" / "policy_output";
    return path;
}

/* 生成适合文件名使用的本地时间戳。 */
std::string local_timestamp_for_filename()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch()).count() % 1000;

    std::tm local_time = {};
    localtime_r(&now_time, &local_time);

    std::ostringstream stream;
    stream << std::put_time(&local_time, "%Y%m%d_%H%M%S")
           << '_' << std::setw(3) << std::setfill('0') << ms;
    return stream.str();
}

void append_indexed_columns(std::ostream& stream,
                            const char* prefix,
                            int count)
{
    for (int i = 0; i < count; ++i) {
        stream << ',' << prefix << '_' << i;
    }
}

void append_indexed_columns_with_frame(std::ostream& stream,
                                       const char* prefix,
                                       int count,
                                       int frame)
{
    for (int i = 0; i < count; ++i) {
        stream << ',' << prefix << '_' << i << "_f" << frame;
    }
}

/**
 * 按 term 优先（每个 term 内帧连续）生成列名。
 * 匹配 fill_term_history / append_term_history 的内存布局：
 * term[0]_f0, ..., term[N-1]_f0, term[0]_f1, ..., term[N-1]_f(frame_stack-1)
 */
void append_term_major_columns(std::ostream& stream,
                               const char* prefix,
                               int count,
                               int frame_stack)
{
    for (int frame = 0; frame < frame_stack; ++frame) {
        for (int i = 0; i < count; ++i) {
            stream << ',' << prefix << '_' << i << "_f" << frame;
        }
    }
}

void write_policy_observation_csv_header(std::ostream& stream)
{
    stream << "frame_index,elapsed_us";
#if POLICY_V3
    constexpr int frame_stack = 15;
    for (int frame = 0; frame < frame_stack; ++frame) {
        stream << ",phase_sin_f" << frame << ",phase_cos_f" << frame;
        append_indexed_columns_with_frame(stream, "velocity_command", 3, frame);
        append_indexed_columns_with_frame(stream, "joint_pos_rel", 12, frame);
        append_indexed_columns_with_frame(stream, "joint_vel_rel", 12, frame);
        append_indexed_columns_with_frame(stream, "last_action", 12, frame);
        append_indexed_columns_with_frame(stream, "body_ang_vel", 3, frame);
        append_indexed_columns_with_frame(stream, "euler", 3, frame);
    }
#else
    constexpr int frame_stack = 5;
    append_term_major_columns(stream, "base_ang_vel", 3, frame_stack);
    append_term_major_columns(stream, "projected_gravity", 3, frame_stack);
    append_term_major_columns(stream, "velocity_command", 3, frame_stack);
    append_term_major_columns(stream, "joint_pos_rel", 12, frame_stack);
    append_term_major_columns(stream, "joint_vel_rel", 12, frame_stack);
    append_term_major_columns(stream, "last_action", 12, frame_stack);
#endif
    stream << '\n';
}

void write_policy_output_csv_header(std::ostream& stream)
{
    stream << "frame_index,elapsed_us";
    append_indexed_columns(stream, "raw_action", 12);
    append_indexed_columns(stream, "target_q_model_rad", 12);
    append_indexed_columns(stream, "target_rad", 12);
    stream << '\n';
}

/* 检查 3 维数组中的数值是否全部为有限值。 */
bool finite_array3(const std::array<double, 3>& values)
{
    return std::isfinite(values[0]) &&
           std::isfinite(values[1]) &&
           std::isfinite(values[2]);
}

/* 检查动态数组中的数值是否全部为有限值。 */
bool finite_vector(const std::vector<double>& values)
{
    return std::all_of(values.begin(), values.end(), [](double value) {
        return std::isfinite(value);
    });
}

/* 检查电机方向配置是否只包含 1 或 -1。 */
bool valid_motor_direction_values(const std::vector<int>& directions)
{
    return std::all_of(directions.begin(), directions.end(), [](int direction) {
        return direction == 1 || direction == -1;
    });
}

/* 获取指定电机从物理坐标到模型坐标的方向符号。 */
int direction_for_motor(const RobotInterfaceConfig& config, int motor_index)
{
    if (config.motor_to_model_direction.empty()) {
        return 1;
    }
    return config.motor_to_model_direction[motor_index];
}
bool index_in_range(int index, int count)
{
    return index >= 0 && index < count;
}

bool ankle_parallel_map_indices_in_range(
    const RobotInterfaceConfig::AnkleParallelMap& ankle_map,
    int count)
{
    return index_in_range(ankle_map.model_pitch_dof, count) &&
           index_in_range(ankle_map.model_roll_dof, count) &&
           index_in_range(ankle_map.upper_motor_index, count) &&
           index_in_range(ankle_map.lower_motor_index, count);
}


/* 判断当前电机控制模式是否走 MIT/PVT setpoint 下发路径。 */
bool is_mit_mode(myactua::ControlMode mode)
{
    return mode == myactua::ControlMode::PVT;
}

}  // namespace

/* 保存外部传入的接口配置，后续由初始化函数按模块使用。 */
RobotInterface::RobotInterface(RobotInterfaceConfig config)
    : config_(std::move(config)) {}

/* 析构时依次释放策略、IMU 和电机资源，保证后台线程退出。 */
RobotInterface::~RobotInterface() {
    unload_policy();
    deinit_imu();
    deinit_motors();
}



/********************************************************************* */
/* 初始化并启动 IMU 读取线程，注册 AHRS 回调以缓存姿态与角速度。 */
bool RobotInterface::initial_and_start_imu() {
    if (imu_initialized_.load()) {
        return true;
    }

    imu::Config_t imu_cfg;
    imu_cfg.device      = config_.imu_device;
    imu_cfg.baudrate    = config_.imu_baudrate;
    imu_cfg.print_imu   = config_.imu_print_imu;
    imu_cfg.print_ahrs  = config_.imu_print_ahrs;

    ahrs_ready_.store(false);
    projected_gravity_valid_ = false;
    imu_reader_ = std::make_unique<imu::IMUReader>();
    imu_reader_->set_imu_callback([this](const imu::IMUData_t& data) {
        (void)data;
    });
    imu_reader_->set_ahrs_callback([this](const imu::AHRSData_t& data) {
        // 策略输入使用 AHRS 的角速度、欧拉角和四元数原始值；Yaw 不做启动零偏。
        std::lock_guard<std::mutex> lock(imu_mutex_);
        ang_vel_[0] = static_cast<double>(data.roll_speed);
        ang_vel_[1] = static_cast<double>(data.pitch_speed);
        ang_vel_[2] = static_cast<double>(data.heading_speed);
        body_ang_vel_ = ang_vel_;

        euler_[0] = static_cast<double>(data.roll);
        euler_[1] = static_cast<double>(data.pitch);
        euler_[2] = static_cast<double>(data.heading);
        
        quat_[0] = static_cast<double>(data.qw);
        quat_[1] = static_cast<double>(data.qx);
        quat_[2] = static_cast<double>(data.qy);
        quat_[3] = static_cast<double>(data.qz);

        projected_gravity_[0] = static_cast<double>(data.projected_gravity_x);
        projected_gravity_[1] = static_cast<double>(data.projected_gravity_y);
        projected_gravity_[2] = static_cast<double>(data.projected_gravity_z);
        projected_gravity_valid_ = data.projected_gravity_valid;
        ahrs_ready_.store(true);
    });

    if (!imu_reader_->start(imu_cfg)) {
        std::cerr << "[RobotInterface] IMU start failed.\n";
        imu_reader_.reset();
        imu_initialized_.store(false);
        return false;
    }

    imu_initialized_.store(true);
    return true;
}

/* 停止 IMU 读取线程并清空 IMU/AHRS 就绪状态。 */
void RobotInterface::deinit_imu() {
    if (imu_reader_) {
        imu_reader_->stop();
    }
    imu_reader_.reset();
    imu_initialized_.store(false);
    ahrs_ready_.store(false);
    projected_gravity_valid_ = false;
}

/* 获取 AHRS 输出的四元数，顺序为 [w, x, y, z]。 */
std::array<double, 4> RobotInterface::get_quat() const {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    return quat_;
}

/* 获取 IMU AHRS 输出的角速度。 */
std::array<double, 3> RobotInterface::get_ang_vel() const {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    return ang_vel_;
}

/* 获取机身坐标系角速度，供策略观测使用。 */
std::array<double, 3> RobotInterface::get_body_ang_vel() const {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    return body_ang_vel_;
}

/* 获取 IMU AHRS 输出的欧拉角。 */
std::array<double, 3> RobotInterface::get_euler() const {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    return euler_;
}

/* 获取投影重力向量；数据无效时返回零向量。 */
std::array<double, 3> RobotInterface::get_projected_gravity() const {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    if (!projected_gravity_valid_) {
        return {0.0, 0.0, 0.0};
    }
    return projected_gravity_;
}


/*************************************************************************** */
/* 初始化 EtherCAT 电机控制器，配置控制模式并启动实时线程。 */
bool RobotInterface::initial_and_start_motors() {
    if (motors_initialized_.load()) {
        return true;
    }

    if (!validate_motor_config()) {
        return false;
    }

    adapter_    = std::make_shared<myactua::EthercatAdapterIGH>();
    controller_ = std::make_unique<myactua::MYACTUA>(adapter_, config_.num_motors);

    std::cout << "[RobotInterface] Connecting EtherCAT on "
              << config_.ethercat_ifname << "...\n";

    if (!controller_->connect(config_.ethercat_ifname.c_str())) {
        std::cerr << "[RobotInterface] EtherCAT connect failed.\n";
        controller_.reset();
        adapter_.reset();
        return false;
    }

    if (!controller_->wait_all_slaves_ready(config_.wait_all_slaves_timeout_ms,
                                            config_.wait_all_slaves_poll_ms)) {
        std::cerr << "[RobotInterface] Not all slaves became ready in timeout.\n";
        controller_.reset();
        adapter_.reset();
        return false;
    }

    /* 设置电机控制模式 */
    for (int i = 0; i < config_.num_motors; ++i) {
        controller_->send_command(
            myactua::ControlCommand::SetMode(config_.motor_control_mode, i));
    }

    /* 根据配置决定是否打印电机信息 */
    if (config_.print_motors_info) {
        controller_->set_print_info(config_.print_motor_ids);
    } else {
        controller_->set_print_info({});
    }

    if (!controller_->start() || !controller_->is_realtime_scheduling_ready()) {
        std::cerr << "[RobotInterface] realtime controller start failed.\n";
        controller_->shutdown();
        controller_.reset();
        adapter_.reset();
        return false;
    }

    controller_->send_command(myactua::ControlCommand::Stop());
    motors_initialized_.store(true);
    motion_enabled_.store(false);

    return true;
}

/* 校验电机数量、方向和 MIT/PVT 控制参数是否满足下发要求。 */
bool RobotInterface::validate_motor_config() const {
    if (config_.num_motors <= 0) {
        std::cerr << "[RobotInterface] num_motors must be positive\n";
        return false;
    }

    if (!config_.motor_to_model_direction.empty()) {
        if (static_cast<int>(config_.motor_to_model_direction.size()) != config_.num_motors) {
            std::cerr << "[RobotInterface] motor_to_model_direction size must be "
                      << config_.num_motors << " or empty, got="
                      << config_.motor_to_model_direction.size() << "\n";
            return false;
        }
        if (!valid_motor_direction_values(config_.motor_to_model_direction)) {
            std::cerr << "[RobotInterface] motor_to_model_direction values must be 1 or -1\n";
            return false;
        }
    }

    if (is_mit_mode(config_.motor_control_mode)) {
        if (static_cast<int>(config_.mit_kp.size()) != config_.num_motors) {
            std::cerr << "[RobotInterface] MIT mode requires mit_kp size="
                      << config_.num_motors << ", got=" << config_.mit_kp.size()
                      << "\n";
            return false;
        }
        if (static_cast<int>(config_.mit_kd.size()) != config_.num_motors) {
            std::cerr << "[RobotInterface] MIT mode requires mit_kd size="
                      << config_.num_motors << ", got=" << config_.mit_kd.size()
                      << "\n";
            return false;
        }
        if (!finite_vector(config_.mit_kp) || !finite_vector(config_.mit_kd)) {
            std::cerr << "[RobotInterface] MIT mode requires finite mit_kp/mit_kd values\n";
            return false;
        }
    }

    return true;
}

/* 停止电机控制器并释放 EtherCAT 相关资源。 */
void RobotInterface::deinit_motors() {
    if (motors_initialized_.load() && controller_) {
        stop_motors(-1);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        controller_->shutdown();
    }

    controller_.reset();
    adapter_.reset();
    motors_initialized_.store(false);
    motion_enabled_.store(false);
}

/* 发送 STOP 指令；slave_index 为 -1 时停止全部电机并关闭运动使能。 */
bool RobotInterface::stop_motors(int slave_index) {
    if (!motors_initialized_.load() || !controller_) {
        return false;
    }
    if (slave_index >= config_.num_motors) {
        std::cerr << "[RobotInterface] stop_motors invalid slave_index="
                  << slave_index << "\n";
        return false;
    }

    controller_->send_command(myactua::ControlCommand::Stop(slave_index));
    if (slave_index < 0) {
        motion_enabled_.store(false);
    }
    return true;
}

/* 发送 RESTART 指令；全部电机重启成功后允许后续动作下发。 */
bool RobotInterface::restart_motors(int slave_index) {
    if (!motors_initialized_.load() || !controller_) {
        return false;
    }
    if (slave_index >= config_.num_motors) {
        std::cerr << "[RobotInterface] restart_motors invalid slave_index="
                  << slave_index << "\n";
        return false;
    }

    // initial_and_start_motors() 里下发的 SetMode 会被紧随其后的 Stop() 清空
    // （Stop 会 clear 离散命令队列），驱动侧 desired.mode 因而停留在默认的 CSP；
    // 而上层仍按 config_ 发送 MIT/PVT 目标，二者错配会让目标值写进驱动不会读取的
    // 字段，表现为"通信/使能全部正常但电机始终不动"。
    // 这里在 Stop 之后补发一次模式设置，确保驱动真正切到期望模式。
    std::cout << "[RobotInterface] restart_motors: SetMode="
              << static_cast<int>(config_.motor_control_mode)
              << ", slave=" << slave_index << std::endl;
    controller_->send_command(
        myactua::ControlCommand::SetMode(config_.motor_control_mode, slave_index));
    controller_->send_command(myactua::ControlCommand::Restart(slave_index));
    if (slave_index < 0 || config_.num_motors == 1) {
        motion_enabled_.store(true);
    }
    return true;
}

bool RobotInterface::is_parallel_ankle_model_dof(int model_index) const {
    if (!config_.ankle_parallel_kinematics_enabled) return false;
    const auto& l = config_.left_ankle_parallel;
    const auto& r = config_.right_ankle_parallel;
    return model_index == l.model_pitch_dof || model_index == l.model_roll_dof ||
           model_index == r.model_pitch_dof || model_index == r.model_roll_dof;
}

int RobotInterface::direct_model_mapping_slot(int model_index) const {
    int slot = 0;
    for (int i = 0; i < model_index; ++i) if (!is_parallel_ankle_model_dof(i)) ++slot;
    return slot;
}

bool RobotInterface::apply_ankle_ik(
    const std::vector<double>& target_q_model_rad, std::vector<double>& target_rad,
    const RobotInterfaceConfig::AnkleParallelMap& map, ankle_motor_ik::Solver& solver,
    double& last_upper, double& last_lower, bool& solved) const {
    if (!ankle_parallel_map_indices_in_range(map, config_.num_motors)) return false;
    double pitch = target_q_model_rad[map.model_pitch_dof];
    double roll = target_q_model_rad[map.model_roll_dof];
    if (!std::isfinite(pitch) || !std::isfinite(roll)) return false;
    const auto clamp_dof = [&](int dof, double value) {
        if (config_.stand_pose_rad.size() != static_cast<size_t>(config_.num_motors) ||
            config_.joint_min_rad.size() != static_cast<size_t>(config_.num_motors) ||
            config_.joint_max_rad.size() != static_cast<size_t>(config_.num_motors)) return value;
        return std::max(config_.stand_pose_rad[dof] + config_.joint_min_rad[dof],
                        std::min(config_.stand_pose_rad[dof] + config_.joint_max_rad[dof], value));
    };
    pitch = clamp_dof(map.model_pitch_dof, pitch);
    roll = clamp_dof(map.model_roll_dof, roll);
    const auto result = solver.solve(roll, pitch);
    double upper = 0.0, lower = 0.0;
    if (result.reachable()) { upper = result.motor1; lower = result.motor2; last_upper = upper; last_lower = lower; solved = true; }
    else if (solved) { upper = last_upper; lower = last_lower; }
    else return false;
    if (!std::isfinite(upper) || !std::isfinite(lower)) return false;
    const double upper_motor_target =
        direction_for_motor(config_, map.upper_motor_index) * upper;
    const double lower_motor_target =
        direction_for_motor(config_, map.lower_motor_index) * lower;
    const int ankle_slot = (&map == &config_.left_ankle_parallel) ? 0 : 2;
    if (!std::isfinite(upper_motor_target) || !std::isfinite(lower_motor_target) ||
        upper_motor_target < config_.ankle_motor_min_rad[ankle_slot] ||
        upper_motor_target > config_.ankle_motor_max_rad[ankle_slot] ||
        lower_motor_target < config_.ankle_motor_min_rad[ankle_slot + 1] ||
        lower_motor_target > config_.ankle_motor_max_rad[ankle_slot + 1]) {
        return false;
    }
    target_rad[map.upper_motor_index] = upper_motor_target;
    target_rad[map.lower_motor_index] = lower_motor_target;
    return true;
}

/* 将模型 DOF 顺序目标关节角转换为电机顺序目标角。 */
bool RobotInterface::build_action_target_rad(
    const std::vector<double>& target_q_model_rad, std::vector<double>& target_rad) const {
    if (static_cast<int>(target_q_model_rad.size()) != config_.num_motors ||
        !finite_vector(target_q_model_rad)) return false;
    target_rad.assign(config_.num_motors, 0.0);
    const bool ankle = config_.ankle_parallel_kinematics_enabled;
    for (int model_index = 0; model_index < config_.num_motors; ++model_index) {
        if (ankle && is_parallel_ankle_model_dof(model_index)) continue;
        const int slot = ankle ? direct_model_mapping_slot(model_index) : model_index;
        if (slot < 0 || slot >= static_cast<int>(config_.model_to_motor_index.size())) return false;
        const int motor_index = config_.model_to_motor_index[slot];
        if (!index_in_range(motor_index, config_.num_motors)) return false;
        double q = target_q_model_rad[model_index];
        if (config_.stand_pose_rad.size() == static_cast<size_t>(config_.num_motors) &&
            config_.joint_min_rad.size() == static_cast<size_t>(config_.num_motors) &&
            config_.joint_max_rad.size() == static_cast<size_t>(config_.num_motors))
            q = std::max(config_.stand_pose_rad[model_index] + config_.joint_min_rad[model_index],
                         std::min(config_.stand_pose_rad[model_index] + config_.joint_max_rad[model_index], q));
        target_rad[motor_index] = direction_for_motor(config_, motor_index) * q;
    }
    if (!ankle) return true;
    return apply_ankle_ik(target_q_model_rad, target_rad, config_.left_ankle_parallel,
                          left_ankle_solver_, left_ankle_last_upper_motor_, left_ankle_last_lower_motor_, left_ankle_solved_) &&
           apply_ankle_ik(target_q_model_rad, target_rad, config_.right_ankle_parallel,
                          right_ankle_solver_, right_ankle_last_upper_motor_, right_ankle_last_lower_motor_, right_ankle_solved_);
}

/* 下发模型 DOF 顺序的目标关节角，内部完成限位、映射和方向转换。 */
bool RobotInterface::apply_action(const std::vector<double>& target_q_model_rad) {
    if (!motors_initialized_.load() || !controller_) {
        return false;
    }
    if (!motion_enabled_.load()) {
        std::cerr << "[RobotInterface] apply_action rejected: motors are stopped. "
                  << "Call restart_motors(-1) first.\n";
        return false;
    }

    std::vector<double> target_rad;
    if (!build_action_target_rad(target_q_model_rad, target_rad)) {
        return false;
    }

    if (is_mit_mode(config_.motor_control_mode)) {
        std::vector<myactua::MitSetpoint> mit_setpoints(config_.num_motors);
        for (int i = 0; i < config_.num_motors; ++i) {
            mit_setpoints[i] = myactua::MitSetpoint(myactua::MYACTUA::rad_to_deg(target_rad[i]),
                                                    0.0,
                                                    0.0,
                                                    config_.mit_kp[i],
                                                    config_.mit_kd[i]);
        }
        controller_->send_command(myactua::ControlCommand::SetMitSetpoints(std::move(mit_setpoints)));
        return true;
    }

    /* 位置控制模式 */
    if (config_.motor_control_mode != myactua::ControlMode::CSP) {
        std::cerr << "[RobotInterface] apply_action supports only MIT/PVT or CSP mode\n";
        return false;
    }

    std::vector<double> target_deg(config_.num_motors, 0.0);
    for (int i = 0; i < config_.num_motors; ++i) {
        target_deg[i] = myactua::MYACTUA::rad_to_deg(target_rad[i]);
    }

    controller_->send_command(myactua::ControlCommand::SetScalarSetpoints(std::move(target_deg)));
    return true;
}

/* 直接按物理电机顺序下发目标角(rad)，跳过模型映射/IK/FK。 */
/* 输入 target_rad 必须已经是物理电机顺序、且已乘过 motor_to_model_direction 的电机目标角。 */
bool RobotInterface::send_motor_rad(const std::vector<double>& target_rad) {
    if (!motors_initialized_.load() || !controller_) {
        return false;
    }
    if (!motion_enabled_.load()) {
        std::cerr << "[RobotInterface] send_motor_rad rejected: motors are stopped. "
                  << "Call restart_motors(-1) first.\n";
        return false;
    }
    if (static_cast<int>(target_rad.size()) != config_.num_motors) {
        std::cerr << "[RobotInterface] send_motor_rad size mismatch. expected="
                  << config_.num_motors << " got=" << target_rad.size() << "\n";
        return false;
    }
    if (!std::all_of(target_rad.begin(), target_rad.end(), [](double value) {
            return std::isfinite(value);
        })) {
        std::cerr << "[RobotInterface] send_motor_rad rejected: target contains NaN or Inf\n";
        return false;
    }

    if (is_mit_mode(config_.motor_control_mode)) {
        std::vector<myactua::MitSetpoint> mit_setpoints(config_.num_motors);
        for (int i = 0; i < config_.num_motors; ++i) {
            mit_setpoints[i] = myactua::MitSetpoint(myactua::MYACTUA::rad_to_deg(target_rad[i]),
                                                    0.0,
                                                    0.0,
                                                    config_.mit_kp[i],
                                                    config_.mit_kd[i]);
        }
        controller_->send_command(myactua::ControlCommand::SetMitSetpoints(std::move(mit_setpoints)));
        return true;
    }

    if (config_.motor_control_mode != myactua::ControlMode::CSP) {
        std::cerr << "[RobotInterface] send_motor_rad supports only MIT/PVT or CSP mode\n";
        return false;
    }

    std::vector<double> target_deg(config_.num_motors, 0.0);
    for (int i = 0; i < config_.num_motors; ++i) {
        target_deg[i] = myactua::MYACTUA::rad_to_deg(target_rad[i]);
    }
    controller_->send_command(myactua::ControlCommand::SetScalarSetpoints(std::move(target_deg)));
    return true;
}

/* 获取当前关节角，单位 rad；控制器未创建时返回零向量。 */
/* 若已调用 calibrate_encoder_zero，返回相对零位的值；否则返回绝对位置。 */
std::vector<double> RobotInterface::get_joint_q() const {
    std::vector<double> q(config_.num_motors, 0.0);
    if (!controller_) {
        return q;
    }
    q = controller_->get_joint_q_rad();
    if (encoder_calibrated_ && encoder_home_offset_.size() == q.size()) {
        for (std::size_t i = 0; i < q.size(); ++i) {
            q[i] -= encoder_home_offset_[i];
        }
    }
    return q;
}

/* 记录当前所有电机的绝对位置为零位；之后 get_joint_q 返回相对该零位的值。 */
void RobotInterface::calibrate_encoder_zero() {
    if (!controller_) {
        return;
    }
    encoder_home_offset_ = controller_->get_joint_q_rad();
    encoder_calibrated_ = true;
    std::cout << "[RobotInterface] encoder zero calibrated at "
              << "motor[0]=" << (encoder_home_offset_.empty() ? 0.0 : encoder_home_offset_[0])
              << " rad\n";
}

/* 获取当前关节速度，单位 rad/s；控制器未创建时返回零向量。 */
std::vector<double> RobotInterface::get_joint_vel() const {
    std::vector<double> dq(config_.num_motors, 0.0);
    if (!controller_) {
        return dq;
    }
    dq = controller_->get_joint_vel_rad_s();
    return dq;
}

/* 获取当前关节力矩原始值；控制器未创建时返回零向量。 */
std::vector<double> RobotInterface::get_joint_tau() const {
    std::vector<double> tau(config_.num_motors, 0.0);
    if (!controller_) {
        return tau;
    }
    tau = controller_->get_joint_tau_raw();
    return tau;
}

/* 获取每个电机的通信状态；控制器未创建时返回全 false。 */
std::vector<bool> RobotInterface::get_motor_comm_status() const {
    std::vector<bool> comm_ok(config_.num_motors, false);
    if (!controller_) {
        return comm_ok;
    }

    const auto status = controller_->get_status();
    for (std::size_t i = 0; i < status.size() && i < comm_ok.size(); ++i) {
        comm_ok[i] = status[i].comm_ok;
    }
    return comm_ok;
}

/* 按平滑插值将关节恢复到 stand_pose_rad 初始姿态。 */
bool RobotInterface::reset_joints() {
    if (!motors_initialized_.load() || !controller_ || !motion_enabled_.load()) return false;
    std::vector<double> target_model = config_.stand_pose_rad;
    if (static_cast<int>(target_model.size()) != config_.num_motors) return false;
    const std::vector<double> q0_motor = get_joint_q();
    std::vector<double> q0_model(config_.num_motors, 0.0);
    for (int i = 0; i < config_.num_motors; ++i) {
        if (config_.ankle_parallel_kinematics_enabled && is_parallel_ankle_model_dof(i)) continue;
        const int slot = config_.ankle_parallel_kinematics_enabled ? direct_model_mapping_slot(i) : i;
        if (slot < 0 || slot >= static_cast<int>(config_.model_to_motor_index.size())) return false;
        const int motor = config_.model_to_motor_index[slot];
        if (!index_in_range(motor, config_.num_motors)) return false;
        q0_model[i] = direction_for_motor(config_, motor) * q0_motor[motor];
    }
    auto fill_ankle = [&](const RobotInterfaceConfig::AnkleParallelMap& map) {
        if (!ankle_parallel_map_indices_in_range(map, config_.num_motors)) return false;
        const double upper = direction_for_motor(config_, map.upper_motor_index) * q0_motor[map.upper_motor_index];
        const double lower = direction_for_motor(config_, map.lower_motor_index) * q0_motor[map.lower_motor_index];
        const auto foot = ankle_motor_fk::solve(upper, lower,
                                                 target_model[map.model_roll_dof],
                                                 target_model[map.model_pitch_dof]);
        if (!foot.reachable) return false;
        q0_model[map.model_pitch_dof] = foot.pitch;
        q0_model[map.model_roll_dof] = foot.roll;
        return true;
    };
    if (config_.ankle_parallel_kinematics_enabled &&
        (!fill_ankle(config_.left_ankle_parallel) || !fill_ankle(config_.right_ankle_parallel))) return false;
    for (int step = 1; step <= 100; ++step) {
        const double alpha = static_cast<double>(step) / 100.0;
        std::vector<double> q(config_.num_motors);
        for (int i = 0; i < config_.num_motors; ++i) q[i] = q0_model[i] * (1.0 - alpha) + target_model[i] * alpha;
        if (!apply_action(q)) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return true;
}

/*************************************************************************** */
/* 加载 TorchScript 策略模型，并重置上一周期动作和观测历史。 */
bool RobotInterface::load_policy() {
    std::lock_guard<std::mutex> lock(policy_mutex_);
    // 加载前先清掉旧 runner 和上一周期动作，避免失败后残留旧策略状态。
    policy_runner_.reset();
    last_action_raw_.fill(0.0F);
    observation_history_.fill(0.0F);
    observation_history_ready_ = false;
    policy_start_time_ = std::chrono::steady_clock::now();
    reset_policy_observation_log();
    // 配置校验必须先于任何按索引访问，避免不完整配置触发越界。
    if (!validate_policy_config()) {
        return false;
    }
    reset_ankle_fk_state();

    auto runner = std::make_unique<TorchPolicyRunner>();
    if (!runner->load(config_.policy_model_path)) {
        std::cerr << "[RobotInterface] load_policy failed: "
                  << runner->last_error() << "\n";
        return false;
    }

    policy_runner_ = std::move(runner);
    return true;
}

/* 检查策略模型路径、映射、缩放、限位和观测参数是否有效。 */
bool RobotInterface::validate_policy_config() const {
    auto fail = [](const std::string& message) {
        std::cerr << "[RobotInterface] invalid policy config: "
                  << message << "\n";
        return false;
    };

    if (config_.num_motors != kPolicyDof) {
        return fail("num_motors must be 12");
    }
    if (config_.policy_model_path.empty()) {
        return fail("policy_model_path is empty");
    }
#if POLICY_V3
    if (!std::isfinite(config_.policy_cycle_time_s) || config_.policy_cycle_time_s <= 0.0) {
        return fail("policy_cycle_time_s must be finite and > 0");
    }
#endif
    if (config_.action_clip.size() != static_cast<std::size_t>(kPolicyDof)) {
        return fail("action_clip must have 12 ranges");
    }
    for (const auto& range : config_.action_clip) {
        if (!std::isfinite(range[0]) || !std::isfinite(range[1]) ||
            range[0] > range[1]) {
            return fail("each action_clip range must be finite and ordered");
        }
    }
    if (config_.stand_pose_rad.size() != static_cast<std::size_t>(kPolicyDof)) {
        return fail("stand_pose_rad must have 12 values");
    }
    if (config_.action_scale.size() != static_cast<std::size_t>(kPolicyDof)) {
        return fail("action_scale must have 12 values");
    }
    if (config_.joint_min_rad.size() != static_cast<std::size_t>(kPolicyDof) ||
        config_.joint_max_rad.size() != static_cast<std::size_t>(kPolicyDof)) {
        return fail("joint_min_rad and joint_max_rad must have 12 values");
    }
    const bool ankle = config_.ankle_parallel_kinematics_enabled;
    if (ankle) {
        if (!ankle_parallel_map_indices_in_range(config_.left_ankle_parallel, kPolicyDof) ||
            !ankle_parallel_map_indices_in_range(config_.right_ankle_parallel, kPolicyDof)) {
            return fail("ankle parallel maps contain an out-of-range index");
        }
        std::array<bool, kPolicyDof> ankle_dofs = {};
        std::array<bool, kPolicyDof> motors = {};
        const auto mark = [](std::array<bool, kPolicyDof>& seen, int index) {
            if (seen[index]) return false;
            seen[index] = true;
            return true;
        };
        const auto& left = config_.left_ankle_parallel;
        const auto& right = config_.right_ankle_parallel;
        if (!mark(ankle_dofs, left.model_pitch_dof) ||
            !mark(ankle_dofs, left.model_roll_dof) ||
            !mark(ankle_dofs, right.model_pitch_dof) ||
            !mark(ankle_dofs, right.model_roll_dof) ||
            !mark(motors, left.upper_motor_index) ||
            !mark(motors, left.lower_motor_index) ||
            !mark(motors, right.upper_motor_index) ||
            !mark(motors, right.lower_motor_index)) {
            return fail("ankle parallel maps must use distinct model DOFs and motors");
        }
        const int direct_count = static_cast<int>(std::count(ankle_dofs.begin(), ankle_dofs.end(), false));
        if (static_cast<int>(config_.model_to_motor_index.size()) != direct_count) {
            return fail("model_to_motor_index must contain only non-ankle mappings");
        }
        for (int motor : config_.model_to_motor_index) {
            if (!index_in_range(motor, kPolicyDof) || !mark(motors, motor)) {
                return fail("direct and ankle mappings must assign each motor once");
            }
        }
        if (std::any_of(motors.begin(), motors.end(), [](bool used) { return !used; })) {
            return fail("direct and ankle mappings must cover all motors");
        }
    } else {
        if (config_.model_to_motor_index.size() != static_cast<std::size_t>(kPolicyDof)) {
            return fail("model_to_motor_index must have 12 values");
        }
        std::array<bool, kPolicyDof> seen = {};
        for (int motor_index : config_.model_to_motor_index) {
            if (!index_in_range(motor_index, kPolicyDof) || seen[motor_index]) {
                return fail("model_to_motor_index must be a 12-motor permutation");
            }
            seen[motor_index] = true;
        }
    }
    if (!config_.motor_to_model_direction.empty() &&
        config_.motor_to_model_direction.size() != static_cast<std::size_t>(kPolicyDof)) {
        return fail("motor_to_model_direction must have 12 values or be empty");
    }
    if (config_.dof_pos_scale.size() != static_cast<std::size_t>(kPolicyDof) ||
        config_.dof_vel_scale.size() != static_cast<std::size_t>(kPolicyDof)) {
        return fail("dof_pos_scale and dof_vel_scale must have 12 values");
    }

    if (!finite_vector(config_.stand_pose_rad) ||
        !finite_vector(config_.action_scale) ||
        !finite_vector(config_.joint_min_rad) ||
        !finite_vector(config_.joint_max_rad) ||
        !finite_vector(config_.dof_pos_scale) ||
        !finite_vector(config_.dof_vel_scale)) {
        return fail("all vector policy values must be finite");
    }
    for (std::size_t i = 0; i < config_.ankle_motor_min_rad.size(); ++i) {
        if (std::isnan(config_.ankle_motor_min_rad[i]) ||
            std::isnan(config_.ankle_motor_max_rad[i]) ||
            config_.ankle_motor_min_rad[i] > config_.ankle_motor_max_rad[i]) {
            return fail("ankle motor limits must be non-NaN and ordered");
        }
    }
    if (!finite_array3(config_.command_scale) ||
        !finite_array3(config_.body_ang_vel_scale)) {
        return fail("command/body_ang_vel scales must be finite");
    }
    if (!valid_motor_direction_values(config_.motor_to_model_direction)) {
        return fail("motor_to_model_direction values must be 1 or -1");
    }
#if POLICY_V3
    if (!finite_array3(config_.euler_scale)) {
        return fail("command/body_ang_vel/euler scales must be finite");
    }
#endif

    for (int i = 0; i < kPolicyDof; ++i) {
        if (config_.action_scale[i] <= 0.0) {
            return fail("action_scale must be > 0 for every model DOF");
        }
        if (config_.joint_min_rad[i] > config_.joint_max_rad[i]) {
            return fail("joint_min_rad must be <= joint_max_rad for every model DOF");
        }
        if (config_.joint_min_rad[i] > 0.0 || config_.joint_max_rad[i] < 0.0) {
            return fail("relative joint limits must include 0 for every model DOF");
        }
    }

    return true;
}

/* 卸载策略模型并清空策略运行状态。 */
void RobotInterface::unload_policy() {
    std::lock_guard<std::mutex> lock(policy_mutex_);
    policy_runner_.reset();
    last_action_raw_.fill(0.0F);
    observation_history_.fill(0.0F);
    observation_history_ready_ = false;
    reset_policy_observation_log();
    reset_policy_output_log();
    left_ankle_solver_.reset();
    right_ankle_solver_.reset();
    left_ankle_solved_ = false;
    right_ankle_solved_ = false;
    left_ankle_last_upper_motor_ = left_ankle_last_lower_motor_ = 0.0;
    right_ankle_last_upper_motor_ = right_ankle_last_lower_motor_ = 0.0;
    reset_ankle_fk_state();
}

/* 保留已加载策略，仅重置动作历史、观测历史和相位起点。 */
void RobotInterface::reset_policy_state() {
    std::lock_guard<std::mutex> lock(policy_mutex_);
    last_action_raw_.fill(0.0F);
    observation_history_.fill(0.0F);
    observation_history_ready_ = false;
    policy_start_time_ = std::chrono::steady_clock::now();
    reset_policy_observation_log();
    reset_policy_output_log();
    left_ankle_solver_.reset();
    right_ankle_solver_.reset();
    left_ankle_solved_ = false;
    right_ankle_solved_ = false;
    left_ankle_last_upper_motor_ = left_ankle_last_lower_motor_ = 0.0;
    right_ankle_last_upper_motor_ = right_ankle_last_lower_motor_ = 0.0;
    reset_ankle_fk_state();
}

/* 关闭当前观测日志，下次成功构建观测时重新创建新文件。 */
void RobotInterface::reset_policy_observation_log() {
    if (policy_observation_log_.is_open()) {
        policy_observation_log_.close();
    }
    policy_observation_frame_index_ = 0;
    policy_observation_log_failed_ = false;
}

/* 确保 policy_observation 日志文件已经打开并写入表头。 */
bool RobotInterface::ensure_policy_observation_log() {
    if (policy_observation_log_failed_) {
        return false;
    }
    if (policy_observation_log_.is_open()) {
        return true;
    }

    try {
        std::filesystem::create_directories(policy_observation_log_dir());
    } catch (const std::filesystem::filesystem_error& error) {
        std::cerr << "[RobotInterface] failed to create policy observation log dir: "
                  << error.what() << "\n";
        policy_observation_log_failed_ = true;
        return false;
    }

    const std::filesystem::path log_path =
        policy_observation_log_dir() /
        ("policy_observation_" + local_timestamp_for_filename() + ".csv");
    policy_observation_log_.open(log_path, std::ios::out);
    if (!policy_observation_log_) {
        std::cerr << "[RobotInterface] failed to open policy observation log: "
                  << log_path << "\n";
        policy_observation_log_failed_ = true;
        return false;
    }

    write_policy_observation_csv_header(policy_observation_log_);
    policy_observation_log_ << std::setprecision(9);
    std::cout << "[RobotInterface] policy observation log: "
              << log_path << "\n";
    return true;
}

/* 记录 build_policy_observation 构建出的完整堆叠观测。 */
void RobotInterface::log_policy_observation_frame(
    const std::array<float, kPolicyObservationSize>& observation) {
    if (!ensure_policy_observation_log()) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            now - policy_start_time_).count();

    policy_observation_log_ << policy_observation_frame_index_ << ','
                            << elapsed_us;
    for (float value : observation) {
        policy_observation_log_ << ',' << value;
    }
    policy_observation_log_ << '\n';
    policy_observation_log_.flush();

    if (!policy_observation_log_) {
        std::cerr << "[RobotInterface] failed to write policy observation log\n";
        policy_observation_log_.close();
        policy_observation_log_failed_ = true;
        return;
    }

    ++policy_observation_frame_index_;
}

/* 关闭当前策略输出日志，下次成功调用时重新创建新文件。 */
void RobotInterface::reset_policy_output_log() {
    if (policy_output_log_.is_open()) {
        policy_output_log_.close();
    }
    policy_output_frame_index_ = 0;
    policy_output_log_failed_ = false;
}

/* 确保策略输出日志文件已经打开并写入表头。 */
bool RobotInterface::ensure_policy_output_log() {
    if (policy_output_log_failed_) {
        return false;
    }
    if (policy_output_log_.is_open()) {
        return true;
    }

    try {
        std::filesystem::create_directories(policy_output_log_dir());
    } catch (const std::filesystem::filesystem_error& error) {
        std::cerr << "[RobotInterface] failed to create policy output log dir: "
                  << error.what() << "\n";
        policy_output_log_failed_ = true;
        return false;
    }

    const std::filesystem::path log_path =
        policy_output_log_dir() /
        ("policy_output_" + local_timestamp_for_filename() + ".csv");
    policy_output_log_.open(log_path, std::ios::out);
    if (!policy_output_log_) {
        std::cerr << "[RobotInterface] failed to open policy output log: "
                  << log_path << "\n";
        policy_output_log_failed_ = true;
        return false;
    }

    write_policy_output_csv_header(policy_output_log_);
    policy_output_log_ << std::setprecision(9);
    std::cout << "[RobotInterface] policy output log: "
              << log_path << "\n";
    return true;
}

/* 记录 policy_step 的 raw_action、target_q_model_rad 和 apply_action 的 target_rad。 */
void RobotInterface::log_policy_output(
    const std::array<float, kPolicyDof>& raw_action,
    const std::vector<double>& target_q_model_rad,
    const std::vector<double>& target_rad) {
    if (!ensure_policy_output_log()) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            now - policy_start_time_).count();

    policy_output_log_ << policy_output_frame_index_ << ','
                       << elapsed_us;
    for (float value : raw_action) {
        policy_output_log_ << ',' << value;
    }
    for (double value : target_q_model_rad) {
        policy_output_log_ << ',' << value;
    }
    for (double value : target_rad) {
        policy_output_log_ << ',' << value;
    }
    policy_output_log_ << '\n';
    policy_output_log_.flush();

    if (!policy_output_log_) {
        std::cerr << "[RobotInterface] failed to write policy output log\n";
        policy_output_log_.close();
        policy_output_log_failed_ = true;
        return;
    }

    ++policy_output_frame_index_;
}

/* 执行一次策略闭环：构建观测、模型推理并下发目标关节角。 */
bool RobotInterface::policy_step(double vx, double vy, double yaw_rate) {
    std::lock_guard<std::mutex> lock(policy_mutex_);

    if (!policy_runner_ || !policy_runner_->is_loaded()) {
        return handle_policy_step_failure("policy is not loaded");
    }

    // 单次策略闭环：状态采样 -> 帧观测 -> 模型推理 -> 目标关节角 -> CSP 下发。
    std::array<float, kPolicyObservationSize> observation = {};
    if (!build_policy_observation(vx, vy, yaw_rate, observation)) {
        return handle_policy_step_failure("failed to build policy observation");
    }

    std::array<float, kPolicyDof> raw_action = {};
    if (!policy_runner_->infer(observation, raw_action)) {
        return handle_policy_step_failure(policy_runner_->last_error());
    }
    if (!std::all_of(raw_action.begin(), raw_action.end(), [](float value) {
            return std::isfinite(value);
        })) {
        return handle_policy_step_failure("policy action contains NaN or Inf");
    }

#if POLICY_V3
    // last_action_raw_ 保存模型原始输出，下一周期作为 v3 单帧观测的 last_action 输入。
    // 训练时 self.actions = clip(actions, -18, 18)，这里保持一致。
#else
    // last_action_raw_ 保存模型原始输出，下一周期进入 policy.pt 观测的 165-224 维。
#endif
    {
        constexpr float kClipActions = 18.0f;
        for (int i = 0; i < kPolicyDof; ++i) {
            raw_action[i] = std::max(-kClipActions, std::min(kClipActions, raw_action[i]));
        }
    }
    last_action_raw_ = raw_action;

    std::vector<double> target_q_model_rad(config_.num_motors, 0.0);
    for (int model_index = 0; model_index < kPolicyDof; ++model_index) {
        // 模型输出先按训练约定截断/缩放，再叠加模型顺序的站立姿态。
        const auto& action_clip = config_.action_clip[model_index];
        const double scaled_action =
            static_cast<double>(raw_action[model_index]) * config_.action_scale[model_index];
        const double clipped_action_offset =
            std::max(action_clip[0], std::min(action_clip[1], scaled_action));

        target_q_model_rad[model_index] =
            config_.stand_pose_rad[model_index] + clipped_action_offset;
    }

    std::vector<double> target_rad;
    if (!build_action_target_rad(target_q_model_rad, target_rad)) {
        return handle_policy_step_failure("failed to build action target rad");
    }

    log_policy_output(raw_action, target_q_model_rad, target_rad);

    if (!apply_action(target_q_model_rad)) {
        return handle_policy_step_failure("failed to apply policy target action");
    }

    return true;
}


/* 首次构建观测历史时，用当前 term 填满历史帧中关于当前的 term。 */
template <std::size_t TermSize, std::size_t ObservationSize>
void fill_term_history(std::array<float, ObservationSize>& history,
                       std::size_t offset,
                       std::size_t frame_stack,
                       const std::array<float, TermSize>& current_term)
{
    for (std::size_t frame = 0; frame < frame_stack; ++frame) {
        std::copy(current_term.begin(), current_term.end(), history.begin() + offset + frame * TermSize);
    }
}

/* 后续构建观测历史时，左移旧帧并把当前 term 追加到末尾。 */
template <std::size_t TermSize, std::size_t ObservationSize>
void append_term_history(std::array<float, ObservationSize>& history,
                         std::size_t offset,
                         std::size_t frame_stack,
                         const std::array<float, TermSize>& current_term)
{
    const std::size_t term_history_size = frame_stack * TermSize;
    /* 左移旧帧 */
    std::copy(history.begin() + offset + TermSize, history.begin() + offset + term_history_size, history.begin() + offset);
    /* 拷贝新帧 */
    std::copy(current_term.begin(), current_term.end(), history.begin() + offset + term_history_size - TermSize);
}

bool RobotInterface::build_joint_observation_terms(
    const std::vector<double>& q, const std::vector<double>& dq,
    std::array<float, kPolicyDof>& pos, std::array<float, kPolicyDof>& vel) {
    if (q.size() != kPolicyDof || dq.size() != kPolicyDof ||
        !finite_vector(q) || !finite_vector(dq)) {
        return false;
    }
    for (int i = 0; i < kPolicyDof; ++i) {
        if (config_.ankle_parallel_kinematics_enabled && is_parallel_ankle_model_dof(i)) continue;
        const int slot = config_.ankle_parallel_kinematics_enabled ? direct_model_mapping_slot(i) : i;
        if (slot < 0 || slot >= static_cast<int>(config_.model_to_motor_index.size())) continue;
        const int motor = config_.model_to_motor_index[slot];
        if (!index_in_range(motor, kPolicyDof)) continue;
        const double direction = direction_for_motor(config_, motor);
        pos[i] = static_cast<float>((direction * q[motor] - config_.stand_pose_rad[i]) * config_.dof_pos_scale[i]);
        vel[i] = static_cast<float>(direction * dq[motor] * config_.dof_vel_scale[i]);
    }
    if (config_.ankle_parallel_kinematics_enabled) {
        if (!apply_ankle_fk_observation(q, dq, config_.left_ankle_parallel,
                                        left_ankle_fk_solver_, pos, vel) ||
            !apply_ankle_fk_observation(q, dq, config_.right_ankle_parallel,
                                        right_ankle_fk_solver_, pos, vel)) {
            return false;
        }
    }
    return true;
}

bool RobotInterface::apply_ankle_fk_observation(
    const std::vector<double>& q,
    const std::vector<double>& dq,
    const RobotInterfaceConfig::AnkleParallelMap& map,
    ankle_motor_fk::Solver& solver,
    std::array<float, kPolicyDof>& pos,
    std::array<float, kPolicyDof>& vel) {
    if (!ankle_parallel_map_indices_in_range(map, kPolicyDof) ||
        map.upper_motor_index >= static_cast<int>(q.size()) ||
        map.lower_motor_index >= static_cast<int>(q.size()) ||
        map.upper_motor_index >= static_cast<int>(dq.size()) ||
        map.lower_motor_index >= static_cast<int>(dq.size())) {
        return false;
    }
    const double upper_direction = direction_for_motor(config_, map.upper_motor_index);
    const double lower_direction = direction_for_motor(config_, map.lower_motor_index);
    const double upper = upper_direction * q[map.upper_motor_index];
    const double lower = lower_direction * q[map.lower_motor_index];
    const double upper_velocity = upper_direction * dq[map.upper_motor_index];
    const double lower_velocity = lower_direction * dq[map.lower_motor_index];
    if (!std::isfinite(upper) || !std::isfinite(lower) ||
        !std::isfinite(upper_velocity) || !std::isfinite(lower_velocity)) {
        return false;
    }

    const auto foot = solver.solve(upper, lower);
    if (!foot.reachable || !std::isfinite(foot.pitch) || !std::isfinite(foot.roll)) {
        return false;
    }
    ankle_motor_jacobian::Result jacobian;
    std::string jacobian_error;
    if (!ankle_motor_jacobian::solve(foot.pitch, foot.roll, upper, lower,
                                     jacobian, jacobian_error)) {
        return false;
    }
    const double pitch_velocity =
        jacobian.virtual_from_motor[0][0] * upper_velocity +
        jacobian.virtual_from_motor[0][1] * lower_velocity;
    const double roll_velocity =
        jacobian.virtual_from_motor[1][0] * upper_velocity +
        jacobian.virtual_from_motor[1][1] * lower_velocity;
    if (!std::isfinite(pitch_velocity) || !std::isfinite(roll_velocity)) {
        return false;
    }
    pos[map.model_pitch_dof] = static_cast<float>(
        (foot.pitch - config_.stand_pose_rad[map.model_pitch_dof]) *
        config_.dof_pos_scale[map.model_pitch_dof]);
    pos[map.model_roll_dof] = static_cast<float>(
        (foot.roll - config_.stand_pose_rad[map.model_roll_dof]) *
        config_.dof_pos_scale[map.model_roll_dof]);
    vel[map.model_pitch_dof] = static_cast<float>(
        pitch_velocity * config_.dof_vel_scale[map.model_pitch_dof]);
    vel[map.model_roll_dof] = static_cast<float>(
        roll_velocity * config_.dof_vel_scale[map.model_roll_dof]);
    return std::isfinite(pos[map.model_pitch_dof]) &&
           std::isfinite(pos[map.model_roll_dof]) &&
           std::isfinite(vel[map.model_pitch_dof]) &&
           std::isfinite(vel[map.model_roll_dof]);
}

void RobotInterface::reset_ankle_fk_state() {
    auto reset = [&](const RobotInterfaceConfig::AnkleParallelMap& map, ankle_motor_fk::Solver& solver) {
        if (!index_in_range(map.model_pitch_dof, kPolicyDof) ||
            !index_in_range(map.model_roll_dof, kPolicyDof) ||
            config_.stand_pose_rad.size() != static_cast<std::size_t>(kPolicyDof)) {
            return;
        }
        solver.reset(config_.stand_pose_rad[map.model_roll_dof], config_.stand_pose_rad[map.model_pitch_dof]);
    };
    if (config_.ankle_parallel_kinematics_enabled) {
        reset(config_.left_ankle_parallel, left_ankle_fk_solver_);
        reset(config_.right_ankle_parallel, right_ankle_fk_solver_);
    }
}

/* 结合速度指令、关节状态和 IMU 数据构建完整策略观测。 */
bool RobotInterface::build_policy_observation(
    double vx,
    double vy,
    double yaw_rate,
    std::array<float, kPolicyObservationSize>& observation) {

    if (!motors_initialized_.load() || !controller_) {
        std::cerr << "[RobotInterface] policy observation rejected: motors are not initialized\n";
        return false;
    }
    if (!imu_initialized_.load()) {
        std::cerr << "[RobotInterface] policy observation rejected: IMU is not initialized\n";
        return false;
    }
    if (!ahrs_ready_.load()) {
        std::cerr << "[RobotInterface] policy observation rejected: AHRS data is not ready\n";
        return false;
    }
    if (!std::isfinite(vx) || !std::isfinite(vy) || !std::isfinite(yaw_rate)) {
        std::cerr << "[RobotInterface] policy observation rejected: command is not finite\n";
        return false;
    }

    const std::vector<double> q_rad    = get_joint_q();     // 拿到关节角度
    const std::vector<double> dq_rad_s = get_joint_vel();   // 拿到关节速度
    if (q_rad.size() != static_cast<std::size_t>(kPolicyDof) || dq_rad_s.size() != static_cast<std::size_t>(kPolicyDof)) {
        std::cerr << "[RobotInterface] policy observation rejected: joint state size mismatch\n";
        return false;
    }

    std::array<double, 3> body_ang_vel = {};
    std::array<double, 3> euler = {};
    std::array<double, 3> projected_gravity_double = {};
    bool projected_gravity_valid = false;
    {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        body_ang_vel = body_ang_vel_;
        euler = euler_;
        projected_gravity_double = projected_gravity_;
        projected_gravity_valid = projected_gravity_valid_;
    }

#if POLICY_V3
    if (!finite_vector(q_rad) || !finite_vector(dq_rad_s) ||
        !finite_array3(body_ang_vel) || !finite_array3(euler)) {
        std::cerr << "[RobotInterface] policy observation rejected: sensor value is not finite\n";
        return false;
    }

    const auto now = std::chrono::steady_clock::now();
    const double elapsed_s =
        std::chrono::duration<double>(now - policy_start_time_).count();
    double phase = 0.0;
    if (!config_.fixed_policy_phase) {
        phase = std::fmod(elapsed_s / config_.policy_cycle_time_s, 1.0);
        if (phase < 0.0) {
            phase += 1.0;
        }
    }
    // 单帧观测顺序必须和训练完全一致：phase、command、q、dq、last_action、IMU。
    std::array<float, kPolicySingleObservationSize> current_observation = {};
    current_observation[0] = static_cast<float>(std::sin(2.0 * kPi * phase));
    current_observation[1] = static_cast<float>(std::cos(2.0 * kPi * phase));

    current_observation[2] = static_cast<float>(vx * config_.command_scale[0]);
    current_observation[3] = static_cast<float>(vy * config_.command_scale[1]);
    current_observation[4] = static_cast<float>(yaw_rate * config_.command_scale[2]);

    std::array<float, kPolicyDof> joint_pos_rel = {};
    std::array<float, kPolicyDof> joint_vel_rel = {};
    if (!build_joint_observation_terms(q_rad, dq_rad_s, joint_pos_rel, joint_vel_rel)) {
        std::cerr << "[RobotInterface] policy observation rejected: ankle FK/Jacobian failed\n";
        return false;
    }
    for (int model_index = 0; model_index < kPolicyDof; ++model_index) {
        current_observation[5 + model_index] = joint_pos_rel[model_index];
        current_observation[17 + model_index] = joint_vel_rel[model_index];
        current_observation[29 + model_index] = last_action_raw_[model_index];
    }

    for (int i = 0; i < 3; ++i) {
        current_observation[41 + i] =
            static_cast<float>(body_ang_vel[i] * config_.body_ang_vel_scale[i]);
        current_observation[44 + i] =
            static_cast<float>(euler[i] * config_.euler_scale[i]);
    }

    /* 训练时历史从全零开始；首次仅把当前帧写入最后一帧。 */
    if (!observation_history_ready_) {
        std::copy(current_observation.begin(),
                  current_observation.end(),
                  observation_history_.end() - kPolicySingleObservationSize);
        observation_history_ready_ = true;
    }
    /* 之后把当前观测帧添加到末尾 */
    else {
        std::copy(observation_history_.begin() + kPolicySingleObservationSize,
                   observation_history_.end(),
                 observation_history_.begin());
        std::copy(current_observation.begin(),
                   current_observation.end(),
                 observation_history_.end() - kPolicySingleObservationSize);
    }

    observation = observation_history_;
    {
        constexpr float kClipObservations = 18.0F;
        for (auto& value : observation) {
            value = std::max(-kClipObservations,
                             std::min(kClipObservations, value));
        }
    }
    log_policy_observation_frame(observation);
    return true;

#else
    if (!finite_vector(q_rad) || !finite_vector(dq_rad_s) ||
        !finite_array3(body_ang_vel) || !finite_array3(projected_gravity_double)) {
        std::cerr << "[RobotInterface] policy observation rejected: sensor value is not finite\n";
        return false;
    }
    if (!projected_gravity_valid) {
        std::cerr << "[RobotInterface] policy observation rejected: projected gravity is invalid\n";
        return false;
    }

    std::array<float, 3> base_ang_vel = {};
    std::array<float, 3> projected_gravity = {};
    std::array<float, 3> velocity_commands = {
        static_cast<float>(vx * config_.command_scale[0]),
        static_cast<float>(vy * config_.command_scale[1]),
        static_cast<float>(yaw_rate * config_.command_scale[2])
    };
    std::array<float, kPolicyDof> joint_pos_rel = {};
    std::array<float, kPolicyDof> joint_vel_rel = {};

    /* 构建缩放后的 角速度 与 projected_gravity */
    for (int i = 0; i < 3; ++i) {
        base_ang_vel[i] = static_cast<float>(body_ang_vel[i] * config_.body_ang_vel_scale[i]);
        projected_gravity[i] = static_cast<float>(projected_gravity_double[i]);
    }

    /* 构建电机相关数据（从模型序号转换得到物理序号） */
    for (int model_index = 0; model_index < kPolicyDof; ++model_index) {
        /* 得到电机模型序号对应的物理序号 */
        const int motor_index = config_.model_to_motor_index[model_index];
        /* 得到对应物理序号电机的坐标系方向 */
        const double direction = static_cast<double>(direction_for_motor(config_, motor_index));

        /* 得到对应电机的 角度(rad) 与 角速度(rad/s) */
        const double q_model  = direction * q_rad[motor_index];
        const double dq_model = direction * dq_rad_s[motor_index];

        /* 转换为模型序号电机的关节位置和速度 */
        joint_pos_rel[model_index] = static_cast<float>((q_model - config_.stand_pose_rad[model_index]) * config_.dof_pos_scale[model_index]);
        joint_vel_rel[model_index] = static_cast<float>(dq_model * config_.dof_vel_scale[model_index]);
    }

    std::array<float, kPolicySingleObservationSize> current_observation = {};
    std::copy(base_ang_vel.begin(), base_ang_vel.end(),
              current_observation.begin());
    std::copy(projected_gravity.begin(), projected_gravity.end(),
              current_observation.begin() + 3);
    std::copy(velocity_commands.begin(), velocity_commands.end(),
              current_observation.begin() + 6);
    std::copy(joint_pos_rel.begin(), joint_pos_rel.end(),
              current_observation.begin() + 9);
    std::copy(joint_vel_rel.begin(), joint_vel_rel.end(),
              current_observation.begin() + 21);
    std::copy(last_action_raw_.begin(), last_action_raw_.end(),
              current_observation.begin() + 33);

    constexpr std::size_t kBaseAngVelOffset = 0;
    constexpr std::size_t kProjectedGravityOffset = 15;
    constexpr std::size_t kVelocityCommandsOffset = 30;
    constexpr std::size_t kJointPosRelOffset = 45;
    constexpr std::size_t kJointVelRelOffset = 105;
    constexpr std::size_t kLastActionOffset = 165;

    if (!observation_history_ready_) {
        fill_term_history(observation_history_, kBaseAngVelOffset,
                          kPolicyFrameStack, base_ang_vel);
        fill_term_history(observation_history_, kProjectedGravityOffset,
                          kPolicyFrameStack, projected_gravity);
        fill_term_history(observation_history_, kVelocityCommandsOffset,
                          kPolicyFrameStack, velocity_commands);
        fill_term_history(observation_history_, kJointPosRelOffset,
                          kPolicyFrameStack, joint_pos_rel);
        fill_term_history(observation_history_, kJointVelRelOffset,
                          kPolicyFrameStack, joint_vel_rel);
        fill_term_history(observation_history_, kLastActionOffset,
                          kPolicyFrameStack, last_action_raw_);
        observation_history_ready_ = true;
    } else {
        append_term_history(observation_history_, kBaseAngVelOffset,
                            kPolicyFrameStack, base_ang_vel);
        append_term_history(observation_history_, kProjectedGravityOffset,
                            kPolicyFrameStack, projected_gravity);
        append_term_history(observation_history_, kVelocityCommandsOffset,
                            kPolicyFrameStack, velocity_commands);
        append_term_history(observation_history_, kJointPosRelOffset,
                            kPolicyFrameStack, joint_pos_rel);
        append_term_history(observation_history_, kJointVelRelOffset,
                            kPolicyFrameStack, joint_vel_rel);
        append_term_history(observation_history_, kLastActionOffset,
                            kPolicyFrameStack, last_action_raw_);
    }

    observation = observation_history_;
    log_policy_observation_frame(observation);
    return true;

#endif
}

/* 处理策略执行失败：打印错误并停止全部电机。 */
bool RobotInterface::handle_policy_step_failure(const std::string& message) {
    std::cerr << "[RobotInterface] policy_step failed: " << message << "\n";
    // 策略链路任何一步失败都停机，避免继续执行上一周期的目标。
    stop_motors(-1);
    return false;
}

}  // namespace inference
