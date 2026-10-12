#pragma once

#include <cmath>

namespace myactua::motor_units {

constexpr double kPi = 3.14159265358979323846;
constexpr double kPosPulsePerRev = 131072.0;
constexpr double kAnklePosPulsePerRev = 2.0 * kPosPulsePerRev;
constexpr double kRawPosToRad = (2.0 * kPi) / kPosPulsePerRev;
constexpr double kAnkleRawPosToRad = (2.0 * kPi) / kAnklePosPulsePerRev;
constexpr double kRawVelToRadPerSec = kRawPosToRad;
constexpr double kAnkleRawVelToRadPerSec = kAnkleRawPosToRad;

constexpr bool is_ankle_motor(int motor_index)
{
    return motor_index == 4 || motor_index == 5 ||
           motor_index == 10 || motor_index == 11;
}

constexpr double raw_pos_to_rad(double raw, int motor_index)
{
    return raw * (is_ankle_motor(motor_index) ? kAnkleRawPosToRad : kRawPosToRad);
}

constexpr double rad_to_raw_pos(double rad, int motor_index)
{
    return rad / (is_ankle_motor(motor_index) ? kAnkleRawPosToRad : kRawPosToRad);
}

constexpr double raw_vel_to_rad_s(double raw, int motor_index)
{
    return raw * (is_ankle_motor(motor_index)
                      ? kAnkleRawVelToRadPerSec
                      : kRawVelToRadPerSec);
}

constexpr double rad_s_to_raw_vel(double rad_s, int motor_index)
{
    return rad_s / (is_ankle_motor(motor_index)
                        ? kAnkleRawVelToRadPerSec
                        : kRawVelToRadPerSec);
}

}  // namespace myactua::motor_units
