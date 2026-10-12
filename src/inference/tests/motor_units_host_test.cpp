#include "motor_units.hpp"

#include <cassert>
#include <cmath>

int main()
{
    const int motor_ids[] = {0, 4, 5, 10, 11};
    for (const int motor_id : motor_ids) {
        const double position = 1.234;
        const double position_raw =
            myactua::motor_units::rad_to_raw_pos(position, motor_id);
        assert(std::abs(myactua::motor_units::raw_pos_to_rad(
                            position_raw, motor_id) - position) < 1e-12);

        const double velocity = -2.5;
        const double velocity_raw =
            myactua::motor_units::rad_s_to_raw_vel(velocity, motor_id);
        assert(std::abs(myactua::motor_units::raw_vel_to_rad_s(
                            velocity_raw, motor_id) - velocity) < 1e-12);
    }

    assert(myactua::motor_units::is_ankle_motor(4));
    assert(myactua::motor_units::is_ankle_motor(11));
    assert(!myactua::motor_units::is_ankle_motor(3));
    return 0;
}
