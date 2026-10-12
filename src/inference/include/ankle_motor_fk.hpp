#pragma once

#include "ankle_motor_ik.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ankle_motor_fk {

struct FootAngles {
    double roll = std::numeric_limits<double>::quiet_NaN();
    double pitch = std::numeric_limits<double>::quiet_NaN();
    bool reachable = false;
};

namespace detail {

using ankle_motor_ik::Geometry;
using ankle_motor_ik::Vec3;

struct Residual {
    double motor1 = std::numeric_limits<double>::quiet_NaN();
    double motor2 = std::numeric_limits<double>::quiet_NaN();
};

inline double distance(const Vec3& lhs, const Vec3& rhs) {
    const Vec3 diff = lhs - rhs;
    return std::sqrt(std::max(0.0, ankle_motor_ik::dot(diff, diff)));
}

inline Residual evaluate_residual(const Vec3& b1, const Vec3& b2, double roll, double pitch) {
    const Vec3 c1 = ankle_motor_ik::foot_rotate_point(Geometry::kC1Zero, roll, pitch);
    const Vec3 c2 = ankle_motor_ik::foot_rotate_point(Geometry::kC2Zero, roll, pitch);

    return {
        distance(b1, c1) - Geometry::kL1,
        distance(b2, c2) - Geometry::kL2,
    };
}

inline bool finite_residual(const Residual& residual) {
    return std::isfinite(residual.motor1) && std::isfinite(residual.motor2);
}

inline double residual_norm(const Residual& residual) {
    return std::hypot(residual.motor1, residual.motor2);
}

inline bool damped_least_squares_step(
    double j11, double j12,
    double j21, double j22,
    const Residual& residual,
    double damping,
    double& delta_roll,
    double& delta_pitch
) {
    const double a = j11 * j11 + j21 * j21 + damping;
    const double b = j11 * j12 + j21 * j22;
    const double d = j12 * j12 + j22 * j22 + damping;
    const double g1 = j11 * residual.motor1 + j21 * residual.motor2;
    const double g2 = j12 * residual.motor1 + j22 * residual.motor2;
    const double det = a * d - b * b;

    if (!std::isfinite(det) || std::abs(det) <= 1e-18) {
        return false;
    }

    delta_roll = (-d * g1 + b * g2) / det;
    delta_pitch = (b * g1 - a * g2) / det;
    return std::isfinite(delta_roll) && std::isfinite(delta_pitch);
}

inline bool solve_from_initial(
    const Vec3& b1,
    const Vec3& b2,
    double initial_roll,
    double initial_pitch,
    double& roll,
    double& pitch
) {
    constexpr int kMaxIterations = 50;
    constexpr int kMaxLineSearchSteps = 12;
    constexpr double kDerivativeStepRad = 1e-6;
    constexpr double kResidualToleranceMm = 1e-4;
    constexpr double kMaxStepRad = 0.35;
    constexpr double kDampingValues[] = {1e-9, 1e-6, 1e-3, 1.0, 100.0};

    roll = initial_roll;
    pitch = initial_pitch;

    Residual residual = evaluate_residual(b1, b2, roll, pitch);
    if (!finite_residual(residual)) {
        return false;
    }

    double norm = residual_norm(residual);
    for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
        if (norm <= kResidualToleranceMm) {
            return true;
        }

        const Residual roll_plus = evaluate_residual(b1, b2, roll + kDerivativeStepRad, pitch);
        const Residual roll_minus = evaluate_residual(b1, b2, roll - kDerivativeStepRad, pitch);
        const Residual pitch_plus = evaluate_residual(b1, b2, roll, pitch + kDerivativeStepRad);
        const Residual pitch_minus = evaluate_residual(b1, b2, roll, pitch - kDerivativeStepRad);
        if (!finite_residual(roll_plus) || !finite_residual(roll_minus) ||
            !finite_residual(pitch_plus) || !finite_residual(pitch_minus)) {
            return false;
        }

        const double j11 = (roll_plus.motor1 - roll_minus.motor1) / (2.0 * kDerivativeStepRad);
        const double j21 = (roll_plus.motor2 - roll_minus.motor2) / (2.0 * kDerivativeStepRad);
        const double j12 = (pitch_plus.motor1 - pitch_minus.motor1) / (2.0 * kDerivativeStepRad);
        const double j22 = (pitch_plus.motor2 - pitch_minus.motor2) / (2.0 * kDerivativeStepRad);

        bool accepted = false;
        double accepted_roll = roll;
        double accepted_pitch = pitch;
        double accepted_norm = norm;

        for (double damping : kDampingValues) {
            double delta_roll = 0.0;
            double delta_pitch = 0.0;
            if (!damped_least_squares_step(
                    j11, j12, j21, j22, residual,
                    damping, delta_roll, delta_pitch)) {
                continue;
            }

            const double step_norm = std::hypot(delta_roll, delta_pitch);
            if (!std::isfinite(step_norm) || step_norm <= 1e-14) {
                continue;
            }
            if (step_norm > kMaxStepRad) {
                const double scale = kMaxStepRad / step_norm;
                delta_roll *= scale;
                delta_pitch *= scale;
            }

            double line_scale = 1.0;
            for (int step = 0; step < kMaxLineSearchSteps; ++step) {
                const double trial_roll = roll + line_scale * delta_roll;
                const double trial_pitch = pitch + line_scale * delta_pitch;
                const Residual trial_residual =
                    evaluate_residual(b1, b2, trial_roll, trial_pitch);
                if (finite_residual(trial_residual)) {
                    const double trial_norm = residual_norm(trial_residual);
                    if (trial_norm < accepted_norm) {
                        accepted = true;
                        accepted_roll = trial_roll;
                        accepted_pitch = trial_pitch;
                        accepted_norm = trial_norm;
                        break;
                    }
                }
                line_scale *= 0.5;
            }

            if (accepted) {
                break;
            }
        }

        if (!accepted) {
            return norm <= kResidualToleranceMm;
        }

        roll = accepted_roll;
        pitch = accepted_pitch;
        residual = evaluate_residual(b1, b2, roll, pitch);
        norm = accepted_norm;
    }

    return norm <= kResidualToleranceMm;
}

}  // namespace detail

class Solver {
public:
    Solver() = default;

    Solver(double initial_roll, double initial_pitch)
        : previous_roll_(std::isfinite(initial_roll) ? initial_roll : 0.0),
          previous_pitch_(std::isfinite(initial_pitch) ? initial_pitch : 0.0) {}

    FootAngles solve(double motor1, double motor2) {
        FootAngles result;
        result.roll = previous_roll_;
        result.pitch = previous_pitch_;

        if (!std::isfinite(motor1) || !std::isfinite(motor2)) {
            return result;
        }

        const detail::Vec3 b1 = ankle_motor_ik::crank_rotate_point(
            detail::Geometry::kA1, detail::Geometry::kV1, motor1);
        const detail::Vec3 b2 = ankle_motor_ik::crank_rotate_point(
            detail::Geometry::kA2, detail::Geometry::kV2, motor2);

        double roll = previous_roll_;
        double pitch = previous_pitch_;
        if (!detail::solve_from_initial(b1, b2, previous_roll_, previous_pitch_, roll, pitch)) {
            return result;
        }

        result.roll = previous_roll_ + ankle_motor_ik::wrap_to_pi(roll - previous_roll_);
        result.pitch = previous_pitch_ + ankle_motor_ik::wrap_to_pi(pitch - previous_pitch_);
        result.reachable = true;

        previous_roll_ = result.roll;
        previous_pitch_ = result.pitch;
        return result;
    }

    double previous_roll() const {
        return previous_roll_;
    }

    double previous_pitch() const {
        return previous_pitch_;
    }

    void reset(double roll = 0.0, double pitch = 0.0) {
        previous_roll_ = std::isfinite(roll) ? roll : 0.0;
        previous_pitch_ = std::isfinite(pitch) ? pitch : 0.0;
    }

private:
    double previous_roll_ = 0.0;
    double previous_pitch_ = 0.0;
};

inline FootAngles solve(
    double motor1,
    double motor2,
    double initial_roll = 0.0,
    double initial_pitch = 0.0
) {
    Solver solver(initial_roll, initial_pitch);
    return solver.solve(motor1, motor2);
}

}  // namespace ankle_motor_fk
