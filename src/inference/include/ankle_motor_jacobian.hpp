#pragma once

#include "ankle_motor_ik.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace ankle_motor_jacobian {

struct Result {
    std::array<std::array<double, 2>, 2> motor_from_virtual{};
    std::array<std::array<double, 2>, 2> virtual_from_motor{};
    double determinant = std::numeric_limits<double>::quiet_NaN();
};

namespace detail {
using ankle_motor_ik::Geometry;
using ankle_motor_ik::Vec3;

inline Vec3 foot_pitch_derivative(const Vec3& point, double roll, double pitch)
{
    const double cr = std::cos(roll), sr = std::sin(roll);
    const double cp = std::cos(pitch), sp = std::sin(pitch);
    const double z1 = sr * point.y + cr * point.z;
    return {-sp * point.x + cp * z1, 0.0, -cp * point.x - sp * z1};
}

inline Vec3 foot_roll_derivative(const Vec3& point, double roll, double pitch)
{
    const double cr = std::cos(roll), sr = std::sin(roll);
    const double cp = std::cos(pitch), sp = std::sin(pitch);
    const double dy = -sr * point.y - cr * point.z;
    const double dz = cr * point.y - sr * point.z;
    return {sp * dz, dy, cp * dz};
}

inline Vec3 crank_motor_derivative(const Vec3& v, double motor)
{
    const double c = std::cos(motor), s = std::sin(motor);
    return {0.0, -s * v.y - c * v.z, c * v.y - s * v.z};
}

inline bool finite_vec(const Vec3& v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

inline bool finite_matrix(const std::array<std::array<double, 2>, 2>& m)
{
    return std::isfinite(m[0][0]) && std::isfinite(m[0][1]) &&
           std::isfinite(m[1][0]) && std::isfinite(m[1][1]);
}
}  // namespace detail

inline bool solve(double pitch, double roll, double motor1, double motor2,
                  Result& result, std::string& error,
                  double singular_tolerance = 1e-10)
{
    if (!std::isfinite(pitch) || !std::isfinite(roll) ||
        !std::isfinite(motor1) || !std::isfinite(motor2) ||
        !std::isfinite(singular_tolerance) || singular_tolerance <= 0.0) {
        error = "ankle jacobian inputs must be finite";
        return false;
    }

    const std::array<detail::Vec3, 2> a{detail::Geometry::kA1, detail::Geometry::kA2};
    const std::array<detail::Vec3, 2> v{detail::Geometry::kV1, detail::Geometry::kV2};
    const std::array<detail::Vec3, 2> c{detail::Geometry::kC1Zero, detail::Geometry::kC2Zero};
    const std::array<double, 2> motors{motor1, motor2};
    Result next;

    for (int i = 0; i < 2; ++i) {
        const auto cp = ankle_motor_ik::foot_rotate_point(c[i], roll, pitch);
        const auto bp = ankle_motor_ik::crank_rotate_point(a[i], v[i], motors[i]);
        const auto residual = cp - bp;
        const auto dp = detail::foot_pitch_derivative(c[i], roll, pitch);
        const auto dr = detail::foot_roll_derivative(c[i], roll, pitch);
        const auto dm = detail::crank_motor_derivative(v[i], motors[i]);
        if (!detail::finite_vec(residual) || !detail::finite_vec(dp) ||
            !detail::finite_vec(dr) || !detail::finite_vec(dm)) {
            error = "ankle jacobian intermediate value is not finite";
            return false;
        }
        const double denominator = -2.0 * ankle_motor_ik::dot(residual, dm);
        if (!std::isfinite(denominator) || std::abs(denominator) <= singular_tolerance) {
            error = "ankle jacobian branch is singular";
            return false;
        }
        next.motor_from_virtual[i][0] =
            -(2.0 * ankle_motor_ik::dot(residual, dp)) / denominator;
        next.motor_from_virtual[i][1] =
            -(2.0 * ankle_motor_ik::dot(residual, dr)) / denominator;
    }

    if (!detail::finite_matrix(next.motor_from_virtual)) {
        error = "ankle motor jacobian is not finite";
        return false;
    }
    const double a00 = next.motor_from_virtual[0][0];
    const double a01 = next.motor_from_virtual[0][1];
    const double a10 = next.motor_from_virtual[1][0];
    const double a11 = next.motor_from_virtual[1][1];
    const double det = a00 * a11 - a01 * a10;
    if (!std::isfinite(det) || std::abs(det) <= singular_tolerance) {
        error = "ankle jacobian matrix is singular";
        return false;
    }
    next.virtual_from_motor = {{{a11 / det, -a01 / det},
                                {-a10 / det, a00 / det}}};
    next.determinant = det;
    if (!detail::finite_matrix(next.virtual_from_motor)) {
        error = "ankle virtual jacobian is not finite";
        return false;
    }
    result = next;
    error.clear();
    return true;
}

}  // namespace ankle_motor_jacobian
