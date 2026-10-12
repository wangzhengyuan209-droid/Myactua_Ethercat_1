#include "ankle_motor_fk.hpp"
#include "ankle_motor_ik.hpp"
#include "ankle_motor_jacobian.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

namespace {

bool close_enough(double lhs, double rhs, double tolerance = 2e-3)
{
    return std::isfinite(lhs) && std::isfinite(rhs) && std::abs(lhs - rhs) <= tolerance;
}

ankle_motor_fk::FootAngles fk_at(double motor1, double motor2,
                                 double roll, double pitch)
{
    ankle_motor_fk::Solver solver(roll, pitch);
    return solver.solve(motor1, motor2);
}

void check_pose(double roll, double pitch)
{
    const auto motors = ankle_motor_ik::solve(roll, pitch);
    assert(motors.reachable());

    ankle_motor_jacobian::Result jacobian;
    std::string error;
    assert(ankle_motor_jacobian::solve(pitch, roll, motors.motor1, motors.motor2,
                                       jacobian, error));

    constexpr double h = 1e-5;
    const auto m1_plus = fk_at(motors.motor1 + h, motors.motor2, roll, pitch);
    const auto m1_minus = fk_at(motors.motor1 - h, motors.motor2, roll, pitch);
    const auto m2_plus = fk_at(motors.motor1, motors.motor2 + h, roll, pitch);
    const auto m2_minus = fk_at(motors.motor1, motors.motor2 - h, roll, pitch);
    assert(m1_plus.reachable && m1_minus.reachable &&
           m2_plus.reachable && m2_minus.reachable);

    const double d_pitch_dm1 = (m1_plus.pitch - m1_minus.pitch) / (2.0 * h);
    const double d_pitch_dm2 = (m2_plus.pitch - m2_minus.pitch) / (2.0 * h);
    const double d_roll_dm1 = (m1_plus.roll - m1_minus.roll) / (2.0 * h);
    const double d_roll_dm2 = (m2_plus.roll - m2_minus.roll) / (2.0 * h);

    assert(close_enough(jacobian.virtual_from_motor[0][0], d_pitch_dm1));
    assert(close_enough(jacobian.virtual_from_motor[0][1], d_pitch_dm2));
    assert(close_enough(jacobian.virtual_from_motor[1][0], d_roll_dm1));
    assert(close_enough(jacobian.virtual_from_motor[1][1], d_roll_dm2));
}

}  // namespace

int main()
{
    check_pose(0.0, 0.0);
    check_pose(0.05, -0.05);

    ankle_motor_jacobian::Result result;
    std::string error;
    assert(!ankle_motor_jacobian::solve(
        std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0,
        result, error));
    assert(!error.empty());

    std::cout << "ankle Jacobian host checks passed\n";
    return 0;
}
