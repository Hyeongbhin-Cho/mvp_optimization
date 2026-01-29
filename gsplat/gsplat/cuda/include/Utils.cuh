#pragma once

#include "Common.h"
#include <ATen/cuda/Atomic.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace gsplat {

namespace cg = cooperative_groups;

///////////////////////////////
// Coordinate Transformations
///////////////////////////////

// Transforms a 3D position from world coordinates to camera coordinates.
// [R | t] is the world-to-camera transformation.
inline __device__ void posW2C(
    const mat3 R,
    const vec3 t,
    const vec3 pW, // Input position in world coordinates
    vec3 &pC       // Output position in camera coordinates
) {
    pC = R * pW + t;
}

// Computes the vector-Jacobian product (VJP) for posW2C.
// This function computes gradients of the transformation with respect to
// inputs.
inline __device__ void posW2C_VJP(
    // Forward inputs
    const mat3 R,
    const vec3 t,
    const vec3 pW, // Input position in world coordinates
    // Gradient output
    const vec3 v_pC, // Gradient of the output position in camera coordinates
    // Gradient inputs (to be accumulated)
    mat3 &v_R, // Gradient w.r.t. R
    vec3 &v_t, // Gradient w.r.t. t
    vec3 &v_pW // Gradient w.r.t. pW
) {
    // Using the rule for differentiating a linear transformation:
    // For D = W * X, G = dL/dD
    // dL/dW = G * X^T, dL/dX = W^T * G
    v_R += glm::outerProduct(v_pC, pW);
    v_t += v_pC;
    v_pW += glm::transpose(R) * v_pC;
}

// Transforms a covariance matrix from world coordinates to camera coordinates.
inline __device__ void covarW2C(
    const mat3 R,
    const mat3 covarW, // Input covariance matrix in world coordinates
    mat3 &covarC       // Output covariance matrix in camera coordinates
) {
    covarC = R * covarW * glm::transpose(R);
}

// Computes the vector-Jacobian product (VJP) for covarW2C.
// This function computes gradients of the transformation with respect to
// inputs.
inline __device__ void covarW2C_VJP(
    // Forward inputs
    const mat3 R,
    const mat3 covarW, // Input covariance matrix in world coordinates
    // Gradient output
    const mat3 v_covarC, // Gradient of the output covariance matrix in camera
                         // coordinates
    // Gradient inputs (to be accumulated)
    mat3 &v_R,     // Gradient w.r.t. rotation matrix
    mat3 &v_covarW // Gradient w.r.t. world covariance matrix
) {
    // Using the rule for differentiating quadratic forms:
    // For D = W * X * W^T, G = dL/dD
    // dL/dX = W^T * G * W
    // dL/dW = G * W * X^T + G^T * W * X
    v_R += v_covarC * R * glm::transpose(covarW) +
           glm::transpose(v_covarC) * R * covarW;
    v_covarW += glm::transpose(R) * v_covarC * R;
}

///////////////////////////////
// Reduce
///////////////////////////////

template <uint32_t DIM, class WarpT>
inline __device__ void warpSum(float *val, WarpT &warp) {
#pragma unroll
    for (uint32_t i = 0; i < DIM; i++) {
        val[i] = cg::reduce(warp, val[i], cg::plus<float>());
    }
}

template <class WarpT> inline __device__ void warpSum(float &val, WarpT &warp) {
    val = cg::reduce(warp, val, cg::plus<float>());
}

template <class WarpT> inline __device__ void warpSum(vec4 &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
    val.z = cg::reduce(warp, val.z, cg::plus<float>());
    val.w = cg::reduce(warp, val.w, cg::plus<float>());
}

template <class WarpT> inline __device__ void warpSum(vec3 &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
    val.z = cg::reduce(warp, val.z, cg::plus<float>());
}

template <class WarpT> inline __device__ void warpSum(vec2 &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
}

template <class WarpT> inline __device__ void warpSum(mat4 &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
    warpSum(val[2], warp);
    warpSum(val[3], warp);
}

template <class WarpT> inline __device__ void warpSum(mat3 &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
    warpSum(val[2], warp);
}

template <class WarpT> inline __device__ void warpSum(mat2 &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
}

template <class WarpT> inline __device__ void warpMax(float &val, WarpT &warp) {
    val = cg::reduce(warp, val, cg::greater<float>());
}

///////////////////////////////
// Quaternion
///////////////////////////////

inline __device__ mat3 quat_to_rotmat(const vec4 quat) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    float inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    return mat3(
        (1.f - 2.f * (y2 + z2)),
        (2.f * (xy + wz)),
        (2.f * (xz - wy)), // 1st col
        (2.f * (xy - wz)),
        (1.f - 2.f * (x2 + z2)),
        (2.f * (yz + wx)), // 2nd col
        (2.f * (xz + wy)),
        (2.f * (yz - wx)),
        (1.f - 2.f * (x2 + y2)) // 3rd col
    );
}

inline __device__ void
quat_to_rotmat_vjp(const vec4 quat, const mat3 v_R, vec4 &v_quat) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    float inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    vec4 v_quat_n = vec4(
        2.f * (x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
               z * (v_R[0][1] - v_R[1][0])),
        2.f *
            (-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
             z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])),
        2.f * (x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
               z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])),
        2.f * (x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
               2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0]))
    );

    vec4 quat_n = vec4(w, x, y, z);
    v_quat += (v_quat_n - glm::dot(v_quat_n, quat_n) * quat_n) * inv_norm;
}

inline __device__ void quat_scale_to_covar_preci(
    const vec4 quat,
    const vec3 scale,
    // optional outputs
    mat3 *covar,
    mat3 *preci
) {
    mat3 R = quat_to_rotmat(quat);
    if (covar != nullptr) {
        // C = R * S * S * Rt
        mat3 S =
            mat3(scale[0], 0.f, 0.f, 0.f, scale[1], 0.f, 0.f, 0.f, scale[2]);
        mat3 M = R * S;
        *covar = M * glm::transpose(M);
    }
    if (preci != nullptr) {
        // P = R * S^-1 * S^-1 * Rt
        mat3 S = mat3(
            1.0f / scale[0],
            0.f,
            0.f,
            0.f,
            1.0f / scale[1],
            0.f,
            0.f,
            0.f,
            1.0f / scale[2]
        );
        mat3 M = R * S;
        *preci = M * glm::transpose(M);
    }
}

inline __device__ void quat_scale_to_covar_vjp(
    // fwd inputs
    const vec4 quat,
    const vec3 scale,
    // precompute
    const mat3 R,
    // grad outputs
    const mat3 v_covar,
    // grad inputs
    vec4 &v_quat,
    vec3 &v_scale
) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    float sx = scale[0], sy = scale[1], sz = scale[2];

    // M = R * S
    mat3 S = mat3(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    mat3 M = R * S;

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    // so
    // for D = M * Mt,
    // df/dM = df/dM + df/dMt = G * M + (Mt * G)t = G * M + Gt * M
    mat3 v_M = (v_covar + glm::transpose(v_covar)) * M;
    mat3 v_R = v_M * S;

    // grad for (quat, scale) from covar
    quat_to_rotmat_vjp(quat, v_R, v_quat);

    v_scale[0] +=
        R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2];
    v_scale[1] +=
        R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2];
    v_scale[2] +=
        R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2];
}

inline __device__ void quat_scale_to_preci_vjp(
    // fwd inputs
    const vec4 quat,
    const vec3 scale,
    // precompute
    const mat3 R,
    // grad outputs
    const mat3 v_preci,
    // grad inputs
    vec4 &v_quat,
    vec3 &v_scale
) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    float sx = 1.0f / scale[0], sy = 1.0f / scale[1], sz = 1.0f / scale[2];

    // M = R * S
    mat3 S = mat3(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    mat3 M = R * S;

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    // so
    // for D = M * Mt,
    // df/dM = df/dM + df/dMt = G * M + (Mt * G)t = G * M + Gt * M
    mat3 v_M = (v_preci + glm::transpose(v_preci)) * M;
    mat3 v_R = v_M * S;

    // grad for (quat, scale) from preci
    quat_to_rotmat_vjp(quat, v_R, v_quat);

    v_scale[0] +=
        -sx * sx *
        (R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2]);
    v_scale[1] +=
        -sy * sy *
        (R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2]);
    v_scale[2] +=
        -sz * sz *
        (R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2]);
}

inline __device__ void quat_scale_to_covar_preci_half(
    const vec4 quat,
    const vec3 scale,
    // optional outputs
    mat3 *covar_half,
    mat3 *preci_half
) {
    mat3 R = quat_to_rotmat(quat);
    if (covar_half != nullptr) {
        // C = R * S
        mat3 S =
            mat3(scale[0], 0.f, 0.f, 0.f, scale[1], 0.f, 0.f, 0.f, scale[2]);
        *covar_half = R * S;
    }
    if (preci_half != nullptr) {
        // P = R * S^-1
        mat3 S = mat3(
            1.0f / scale[0],
            0.f,
            0.f,
            0.f,
            1.0f / scale[1],
            0.f,
            0.f,
            0.f,
            1.0f / scale[2]
        );
        *preci_half = R * S;
    }
}

inline __device__ void quat_scale_to_preci_half_vjp(
    // fwd inputs
    const vec4 quat,
    const vec3 scale,
    // precompute
    const mat3 R,
    // grad outputs
    const mat3 v_M, // M is perci_half
    // grad inputs
    vec4 &v_quat,
    vec3 &v_scale
) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    float sx = 1.0f / scale[0], sy = 1.0f / scale[1], sz = 1.0f / scale[2];

    // M = R * S
    mat3 S = mat3(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    mat3 v_R = v_M * S;

    // grad for (quat, scale) from preci
    quat_to_rotmat_vjp(quat, v_R, v_quat);

    v_scale[0] +=
        -sx * sx *
        (R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2]);
    v_scale[1] +=
        -sy * sy *
        (R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2]);
    v_scale[2] +=
        -sz * sz *
        (R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2]);
}

///////////////////////////////
// Misc
///////////////////////////////

inline __device__ void
inverse_vjp(const mat2 Minv, const mat2 v_Minv, mat2 &v_M) {
    // P = M^-1
    // df/dM = -P * df/dP * P
    v_M += -Minv * v_Minv * Minv;
}

inline __device__ float
add_blur(const float eps2d, mat2 &covar, float &compensation) {
    float det_orig = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    covar[0][0] += eps2d;
    covar[1][1] += eps2d;
    float det_blur = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    compensation = sqrt(max(0.f, det_orig / det_blur));
    return det_blur;
}

inline __device__ void add_blur_vjp(
    const float eps2d,
    const mat2 conic_blur,
    const float compensation,
    const float v_compensation,
    mat2 &v_covar
) {
    // comp = sqrt(det(covar) / det(covar_blur))

    // d [det(M)] / d M = adj(M)
    // d [det(M + aI)] / d M  = adj(M + aI) = adj(M) + a * I
    // d [det(M) / det(M + aI)] / d M
    // = (det(M + aI) * adj(M) - det(M) * adj(M + aI)) / (det(M + aI))^2
    // = adj(M) / det(M + aI) - adj(M + aI) / det(M + aI) * comp^2
    // = (adj(M) - adj(M + aI) * comp^2) / det(M + aI)
    // given that adj(M + aI) = adj(M) + a * I
    // = (adj(M + aI) - aI - adj(M + aI) * comp^2) / det(M + aI)
    // given that adj(M) / det(M) = inv(M)
    // = (1 - comp^2) * inv(M + aI) - aI / det(M + aI)
    // given det(inv(M)) = 1 / det(M)
    // = (1 - comp^2) * inv(M + aI) - aI * det(inv(M + aI))
    // = (1 - comp^2) * conic_blur - aI * det(conic_blur)

    float det_conic_blur = conic_blur[0][0] * conic_blur[1][1] -
                           conic_blur[0][1] * conic_blur[1][0];
    float v_sqr_comp = v_compensation * 0.5 / (compensation + 1e-6);
    float one_minus_sqr_comp = 1 - compensation * compensation;
    v_covar[0][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][0] -
                                   eps2d * det_conic_blur);
    v_covar[0][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][1]);
    v_covar[1][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][0]);
    v_covar[1][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][1] -
                                   eps2d * det_conic_blur);
}

///////////////////////////////
// Projection Related
///////////////////////////////

inline __device__ void ortho_proj(
    // inputs
    const vec3 mean3d,
    const mat3 cov3d,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2 &cov2d,
    vec2 &mean2d
) {
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    // mat3x2 is 3 columns x 2 rows.
    mat3x2 J = mat3x2(
        fx,
        0.f, // 1st column
        0.f,
        fy, // 2nd column
        0.f,
        0.f // 3rd column
    );
    cov2d = J * cov3d * glm::transpose(J);
    mean2d = vec2({fx * x + cx, fy * y + cy});
}

inline __device__ void ortho_proj_vjp(
    // fwd inputs
    const vec3 mean3d,
    const mat3 cov3d,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2 v_cov2d,
    const vec2 v_mean2d,
    // grad inputs
    vec3 &v_mean3d,
    mat3 &v_cov3d
) {
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    // mat3x2 is 3 columns x 2 rows.
    mat3x2 J = mat3x2(
        fx,
        0.f, // 1st column
        0.f,
        fy, // 2nd column
        0.f,
        0.f // 3rd column
    );

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    // df/dx = fx * df/dpixx
    // df/dy = fy * df/dpixy
    // df/dz = 0
    v_mean3d += vec3(fx * v_mean2d[0], fy * v_mean2d[1], 0.f);
}

inline __device__ void persp_proj(
    // inputs
    const vec3 mean3d,
    const mat3 cov3d,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2 &cov2d,
    vec2 &mean2d
) {
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    float tan_fovx = 0.5f * width / fx;
    float tan_fovy = 0.5f * height / fy;
    float lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    float lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    float rz = 1.f / z;
    float rz2 = rz * rz;
    float tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    float ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    mat3x2 J = mat3x2(
        fx * rz,
        0.f, // 1st column
        0.f,
        fy * rz, // 2nd column
        -fx * tx * rz2,
        -fy * ty * rz2 // 3rd column
    );
    cov2d = J * cov3d * glm::transpose(J);
    mean2d = vec2({fx * x * rz + cx, fy * y * rz + cy});
}

inline __device__ void persp_proj_vjp(
    // fwd inputs
    const vec3 mean3d,
    const mat3 cov3d,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2 v_cov2d,
    const vec2 v_mean2d,
    // grad inputs
    vec3 &v_mean3d,
    mat3 &v_cov3d
) {
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    float tan_fovx = 0.5f * width / fx;
    float tan_fovy = 0.5f * height / fy;
    float lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    float lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    float rz = 1.f / z;
    float rz2 = rz * rz;
    float tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    float ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    mat3x2 J = mat3x2(
        fx * rz,
        0.f, // 1st column
        0.f,
        fy * rz, // 2nd column
        -fx * tx * rz2,
        -fy * ty * rz2 // 3rd column
    );

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    // df/dx = fx * rz * df/dpixx
    // df/dy = fy * rz * df/dpixy
    // df/dz = - fx * mean.x * rz2 * df/dpixx - fy * mean.y * rz2 * df/dpixy
    v_mean3d += vec3(
        fx * rz * v_mean2d[0],
        fy * rz * v_mean2d[1],
        -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * rz2
    );

    // df/dx = -fx * rz2 * df/dJ_02
    // df/dy = -fy * rz2 * df/dJ_12
    // df/dz = -fx * rz2 * df/dJ_00 - fy * rz2 * df/dJ_11
    //         + 2 * fx * tx * rz3 * df/dJ_02 + 2 * fy * ty * rz3
    float rz3 = rz2 * rz;
    mat3x2 v_J = v_cov2d * J * glm::transpose(cov3d) +
                 glm::transpose(v_cov2d) * J * cov3d;

    // fov clipping
    if (x * rz <= lim_x_pos && x * rz >= -lim_x_neg) {
        v_mean3d.x += -fx * rz2 * v_J[2][0];
    } else {
        v_mean3d.z += -fx * rz3 * v_J[2][0] * tx;
    }
    if (y * rz <= lim_y_pos && y * rz >= -lim_y_neg) {
        v_mean3d.y += -fy * rz2 * v_J[2][1];
    } else {
        v_mean3d.z += -fy * rz3 * v_J[2][1] * ty;
    }
    v_mean3d.z += -fx * rz2 * v_J[0][0] - fy * rz2 * v_J[1][1] +
                  2.f * fx * tx * rz3 * v_J[2][0] +
                  2.f * fy * ty * rz3 * v_J[2][1];
}

inline __device__ void fisheye_proj(
    // inputs
    const vec3 mean3d,
    const mat3 cov3d,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2 &cov2d,
    vec2 &mean2d
) {
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    float eps = 0.0000001f;
    float xy_len = glm::length(glm::vec2({x, y})) + eps;
    float theta = glm::atan(xy_len, z + eps);
    mean2d = vec2({x * fx * theta / xy_len + cx, y * fy * theta / xy_len + cy});

    float x2 = x * x + eps;
    float y2 = y * y;
    float xy = x * y;
    float x2y2 = x2 + y2;
    float x2y2z2_inv = 1.f / (x2y2 + z * z);

    float b = glm::atan(xy_len, z) / xy_len / x2y2;
    float a = z * x2y2z2_inv / (x2y2);
    mat3x2 J = mat3x2(
        fx * (x2 * a + y2 * b),
        fy * xy * (a - b),
        fx * xy * (a - b),
        fy * (y2 * a + x2 * b),
        -fx * x * x2y2z2_inv,
        -fy * y * x2y2z2_inv
    );
    cov2d = J * cov3d * glm::transpose(J);
}

inline __device__ void fisheye_proj_vjp(
    // fwd inputs
    const vec3 mean3d,
    const mat3 cov3d,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2 v_cov2d,
    const vec2 v_mean2d,
    // grad inputs
    vec3 &v_mean3d,
    mat3 &v_cov3d
) {
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    const float eps = 0.0000001f;
    float x2 = x * x + eps;
    float y2 = y * y;
    float xy = x * y;
    float x2y2 = x2 + y2;
    float len_xy = length(glm::vec2({x, y})) + eps;
    const float x2y2z2 = x2y2 + z * z;
    float x2y2z2_inv = 1.f / x2y2z2;
    float b = glm::atan(len_xy, z) / len_xy / x2y2;
    float a = z * x2y2z2_inv / (x2y2);
    v_mean3d += vec3(
        fx * (x2 * a + y2 * b) * v_mean2d[0] + fy * xy * (a - b) * v_mean2d[1],
        fx * xy * (a - b) * v_mean2d[0] + fy * (y2 * a + x2 * b) * v_mean2d[1],
        -fx * x * x2y2z2_inv * v_mean2d[0] - fy * y * x2y2z2_inv * v_mean2d[1]
    );

    const float theta = glm::atan(len_xy, z);
    const float J_b = theta / len_xy / x2y2;
    const float J_a = z * x2y2z2_inv / (x2y2);
    // mat3x2 is 3 columns x 2 rows.
    mat3x2 J = mat3x2(
        fx * (x2 * J_a + y2 * J_b),
        fy * xy * (J_a - J_b), // 1st column
        fx * xy * (J_a - J_b),
        fy * (y2 * J_a + x2 * J_b), // 2nd column
        -fx * x * x2y2z2_inv,
        -fy * y * x2y2z2_inv // 3rd column
    );
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    mat3x2 v_J = v_cov2d * J * glm::transpose(cov3d) +
                 glm::transpose(v_cov2d) * J * cov3d;
    float l4 = x2y2z2 * x2y2z2;

    float E = -l4 * x2y2 * theta + x2y2z2 * x2y2 * len_xy * z;
    float F = 3 * l4 * theta - 3 * x2y2z2 * len_xy * z - 2 * x2y2 * len_xy * z;

    float A = x * (3 * E + x2 * F);
    float B = y * (E + x2 * F);
    float C = x * (E + y2 * F);
    float D = y * (3 * E + y2 * F);

    float S1 = x2 - y2 - z * z;
    float S2 = y2 - x2 - z * z;
    float inv1 = x2y2z2_inv * x2y2z2_inv;
    float inv2 = inv1 / (x2y2 * x2y2 * len_xy);

    float dJ_dx00 = fx * A * inv2;
    float dJ_dx01 = fx * B * inv2;
    float dJ_dx02 = fx * S1 * inv1;
    float dJ_dx10 = fy * B * inv2;
    float dJ_dx11 = fy * C * inv2;
    float dJ_dx12 = 2.f * fy * xy * inv1;

    float dJ_dy00 = dJ_dx01;
    float dJ_dy01 = fx * C * inv2;
    float dJ_dy02 = 2.f * fx * xy * inv1;
    float dJ_dy10 = dJ_dx11;
    float dJ_dy11 = fy * D * inv2;
    float dJ_dy12 = fy * S2 * inv1;

    float dJ_dz00 = dJ_dx02;
    float dJ_dz01 = dJ_dy02;
    float dJ_dz02 = 2.f * fx * x * z * inv1;
    float dJ_dz10 = dJ_dx12;
    float dJ_dz11 = dJ_dy12;
    float dJ_dz12 = 2.f * fy * y * z * inv1;

    float dL_dtx_raw = dJ_dx00 * v_J[0][0] + dJ_dx01 * v_J[1][0] +
                       dJ_dx02 * v_J[2][0] + dJ_dx10 * v_J[0][1] +
                       dJ_dx11 * v_J[1][1] + dJ_dx12 * v_J[2][1];
    float dL_dty_raw = dJ_dy00 * v_J[0][0] + dJ_dy01 * v_J[1][0] +
                       dJ_dy02 * v_J[2][0] + dJ_dy10 * v_J[0][1] +
                       dJ_dy11 * v_J[1][1] + dJ_dy12 * v_J[2][1];
    float dL_dtz_raw = dJ_dz00 * v_J[0][0] + dJ_dz01 * v_J[1][0] +
                       dJ_dz02 * v_J[2][0] + dJ_dz10 * v_J[0][1] +
                       dJ_dz11 * v_J[1][1] + dJ_dz12 * v_J[2][1];
    v_mean3d.x += dL_dtx_raw;
    v_mean3d.y += dL_dty_raw;
    v_mean3d.z += dL_dtz_raw;
}

inline __device__ vec3 safe_normalize(vec3 v) {
    const float l = v.x * v.x + v.y * v.y + v.z * v.z;
    return l > 0.0f ? (v * rsqrtf(l)) : v;
}

inline __device__ vec3 safe_normalize_bw(const vec3 &v, const vec3 &d_out) {
    const float l = v.x * v.x + v.y * v.y + v.z * v.z;
    if (l > 0.0f) {
        const float il = rsqrtf(l);
        const float il3 = (il * il * il);
        return il * d_out - il3 * glm::dot(d_out, v) * v;
    }
    return d_out;
}

template <typename scalar_t>
inline __device__ float spherical_harmonics_opacity(
    const int degree, 
    const vec3 &dir, 
    const scalar_t *coeffs
) {
    float sum = 0.2820947917738781f * coeffs[0];

    if (degree >= 1) {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;

        sum += 0.48860251190292f * (-y * coeffs[1] + z * coeffs[2] - x * coeffs[3]);

        if (degree >= 2) {
            float z2 = z * z;

            float fTmp0B = -1.092548430592079f * z;
            float fC1 = x * x - y * y;
            float fS1 = 2.f * x * y;
            float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
            float pSH7 = fTmp0B * x;
            float pSH5 = fTmp0B * y;
            float pSH8 = 0.5462742152960395f * fC1;
            float pSH4 = 0.5462742152960395f * fS1;

            sum += pSH4 * coeffs[4] + pSH5 * coeffs[5] +
                   pSH6 * coeffs[6] + pSH7 * coeffs[7] +
                   pSH8 * coeffs[8];

            if (degree >= 3) {
                float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
                float fTmp1B = 1.445305721320277f * z;
                float fC2 = x * fC1 - y * fS1;
                float fS2 = x * fS1 + y * fC1;
                float pSH12 =
                    z * (1.865881662950577f * z2 - 1.119528997770346f);
                float pSH13 = fTmp0C * x;
                float pSH11 = fTmp0C * y;
                float pSH14 = fTmp1B * fC1;
                float pSH10 = fTmp1B * fS1;
                float pSH15 = -0.5900435899266435f * fC2;
                float pSH9 = -0.5900435899266435f * fS2;

                sum += pSH9 * coeffs[9] + pSH10 * coeffs[10] +
                       pSH11 * coeffs[11] + pSH12 * coeffs[12] +
                       pSH13 * coeffs[13] + pSH14 * coeffs[14] +
                       pSH15 * coeffs[15];

                if (degree >= 4) {
                    float fTmp0D =
                        z * (-4.683325804901025f * z2 + 2.007139630671868f);
                    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
                    float fTmp2B = -1.770130769779931f * z;
                    float fC3 = x * fC2 - y * fS2;
                    float fS3 = x * fS2 + y * fC2;
                    float pSH20 =
                        (1.984313483298443f * z * pSH12 -
                         1.006230589874905f * pSH6);
                    float pSH21 = fTmp0D * x;
                    float pSH19 = fTmp0D * y;
                    float pSH22 = fTmp1C * fC1;
                    float pSH18 = fTmp1C * fS1;
                    float pSH23 = fTmp2B * fC2;
                    float pSH17 = fTmp2B * fS2;
                    float pSH24 = 0.6258357354491763f * fC3;
                    float pSH16 = 0.6258357354491763f * fS3;

                    sum += pSH16 * coeffs[16] +
                           pSH17 * coeffs[17] +
                           pSH18 * coeffs[18] +
                           pSH19 * coeffs[19] +
                           pSH20 * coeffs[20] +
                           pSH21 * coeffs[21] +
                           pSH22 * coeffs[22] +
                           pSH23 * coeffs[23] +
                           pSH24 * coeffs[24];
                }
            }
        }
    }

    // sigmoid
    return 1.0f / (1.0f + expf(-sum));
}

template <typename scalar_t>
inline __device__ void spherical_harmonics_opacity_vjp(
    // fwd inputs
    const int degree,
    const vec3 &dir, 
    const scalar_t *coeffs, 
    // fwd outputs
    const float opacity,
    // grad outputs
    const float v_opacity,
    // grad intputs
    scalar_t *v_coeffs,
    vec3 *v_dir
) {
    // graident of sigmoid
    float v_sh_sum = v_opacity * (opacity * (1.0f - opacity));

    gpuAtomicAdd(&v_coeffs[0], (scalar_t)(v_sh_sum * 0.2820947917738781f));

    if (degree < 1) return;

    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    float v_x = 0.f, v_y = 0.f, v_z = 0.f;
    float pSH123 = v_sh_sum * 0.48860251190292f;

    gpuAtomicAdd(&v_coeffs[1], (scalar_t)(pSH123 * -y));
    gpuAtomicAdd(&v_coeffs[2], (scalar_t)(pSH123 * z));
    gpuAtomicAdd(&v_coeffs[3], (scalar_t)(pSH123 * -x));

    if (v_dir != nullptr) {
        v_x += -pSH123 * coeffs[3];
        v_y += -pSH123 * coeffs[1];
        v_z += pSH123 * coeffs[2];
    }

    if (degree < 2) {
        if (v_dir != nullptr) {
            v_dir->x = v_x;
            v_dir->y = v_y;
            v_dir->z = v_z;
        }
        return;
    }

    float z2 = z * z;
    float fTmp0B = -1.092548430592079f * z;
    float fC1 = x * x - y * y;
    float fS1 = 2.f * x * y;
    float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
    float pSH7 = fTmp0B * x;
    float pSH5 = fTmp0B * y;
    float pSH8 = 0.5462742152960395f * fC1;
    float pSH4 = 0.5462742152960395f * fS1;
    gpuAtomicAdd(&v_coeffs[4], (scalar_t)(v_sh_sum * pSH4));
    gpuAtomicAdd(&v_coeffs[5], (scalar_t)(v_sh_sum * pSH5));
    gpuAtomicAdd(&v_coeffs[6], (scalar_t)(v_sh_sum * pSH6));
    gpuAtomicAdd(&v_coeffs[7], (scalar_t)(v_sh_sum * pSH7));
    gpuAtomicAdd(&v_coeffs[8], (scalar_t)(v_sh_sum * pSH8));

    float fTmp0B_z, fC1_x, fC1_y, fS1_x, fS1_y, pSH6_z, pSH7_x, pSH7_z, pSH5_y,
        pSH5_z, pSH8_x, pSH8_y, pSH4_x, pSH4_y;
    if (v_dir != nullptr) {
        fTmp0B_z = -1.092548430592079f;
        fC1_x = 2.f * x;
        fC1_y = -2.f * y;
        fS1_x = 2.f * y;
        fS1_y = 2.f * x;
        pSH6_z = 2.f * 0.9461746957575601f * z;
        pSH7_x = fTmp0B;
        pSH7_z = fTmp0B_z * x;
        pSH5_y = fTmp0B;
        pSH5_z = fTmp0B_z * y;
        pSH8_x = 0.5462742152960395f * fC1_x;
        pSH8_y = 0.5462742152960395f * fC1_y;
        pSH4_x = 0.5462742152960395f * fS1_x;
        pSH4_y = 0.5462742152960395f * fS1_y;

        v_x += v_sh_sum *
               (pSH4_x * coeffs[4] + pSH8_x * coeffs[8] +
                pSH7_x * coeffs[7]);
        v_y += v_sh_sum *
               (pSH4_y * coeffs[4] + pSH8_y * coeffs[8] +
                pSH5_y * coeffs[5]);
        v_z += v_sh_sum *
               (pSH6_z * coeffs[6] + pSH7_z * coeffs[7] +
                pSH5_z * coeffs[5]);
    }

    if (degree < 3) {
        if (v_dir != nullptr) {
            v_dir->x = v_x;
            v_dir->y = v_y;
            v_dir->z = v_z;
        }
        return;
    }

    float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    float fTmp1B = 1.445305721320277f * z;
    float fC2 = x * fC1 - y * fS1;
    float fS2 = x * fS1 + y * fC1;
    float pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
    float pSH13 = fTmp0C * x;
    float pSH11 = fTmp0C * y;
    float pSH14 = fTmp1B * fC1;
    float pSH10 = fTmp1B * fS1;
    float pSH15 = -0.5900435899266435f * fC2;
    float pSH9 = -0.5900435899266435f * fS2;
    gpuAtomicAdd(&v_coeffs[9], (scalar_t)(v_sh_sum * pSH9));
    gpuAtomicAdd(&v_coeffs[10], (scalar_t)(v_sh_sum * pSH10));
    gpuAtomicAdd(&v_coeffs[11], (scalar_t)(v_sh_sum * pSH11));
    gpuAtomicAdd(&v_coeffs[12], (scalar_t)(v_sh_sum * pSH12));
    gpuAtomicAdd(&v_coeffs[13], (scalar_t)(v_sh_sum * pSH13));
    gpuAtomicAdd(&v_coeffs[14], (scalar_t)(v_sh_sum * pSH14));
    gpuAtomicAdd(&v_coeffs[15], (scalar_t)(v_sh_sum * pSH15));


    float fTmp0C_z, fTmp1B_z, fC2_x, fC2_y, fS2_x, fS2_y, pSH12_z, pSH13_x,
        pSH13_z, pSH11_y, pSH11_z, pSH14_x, pSH14_y, pSH14_z, pSH10_x, pSH10_y,
        pSH10_z, pSH15_x, pSH15_y, pSH9_x, pSH9_y;
    if (v_dir != nullptr) {
        fTmp0C_z = -2.285228997322329f * 2.f * z;
        fTmp1B_z = 1.445305721320277f;
        fC2_x = fC1 + x * fC1_x - y * fS1_x;
        fC2_y = x * fC1_y - fS1 - y * fS1_y;
        fS2_x = fS1 + x * fS1_x + y * fC1_x;
        fS2_y = x * fS1_y + fC1 + y * fC1_y;
        pSH12_z = 3.f * 1.865881662950577f * z2 - 1.119528997770346f;
        pSH13_x = fTmp0C;
        pSH13_z = fTmp0C_z * x;
        pSH11_y = fTmp0C;
        pSH11_z = fTmp0C_z * y;
        pSH14_x = fTmp1B * fC1_x;
        pSH14_y = fTmp1B * fC1_y;
        pSH14_z = fTmp1B_z * fC1;
        pSH10_x = fTmp1B * fS1_x;
        pSH10_y = fTmp1B * fS1_y;
        pSH10_z = fTmp1B_z * fS1;
        pSH15_x = -0.5900435899266435f * fC2_x;
        pSH15_y = -0.5900435899266435f * fC2_y;
        pSH9_x = -0.5900435899266435f * fS2_x;
        pSH9_y = -0.5900435899266435f * fS2_y;

        v_x += v_sh_sum *
               (pSH9_x * coeffs[9] + pSH15_x * coeffs[15] +
                pSH10_x * coeffs[10] + pSH14_x * coeffs[14] +
                pSH13_x * coeffs[13]);

        v_y += v_sh_sum *
               (pSH9_y * coeffs[9] + pSH15_y * coeffs[15] +
                pSH10_y * coeffs[10] + pSH14_y * coeffs[14] +
                pSH11_y * coeffs[11]);

        v_z += v_sh_sum *
               (pSH12_z * coeffs[12] + pSH13_z * coeffs[13] +
                pSH11_z * coeffs[11] + pSH14_z * coeffs[14] +
                pSH10_z * coeffs[10]);
    }

    if (degree < 4) {
        if (v_dir != nullptr) {
            v_dir->x = v_x;
            v_dir->y = v_y;
            v_dir->z = v_z;
        }
        return;
    }

    float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    float fTmp2B = -1.770130769779931f * z;
    float fC3 = x * fC2 - y * fS2;
    float fS3 = x * fS2 + y * fC2;
    float pSH20 = (1.984313483298443f * z * pSH12 + -1.006230589874905f * pSH6);
    float pSH21 = fTmp0D * x;
    float pSH19 = fTmp0D * y;
    float pSH22 = fTmp1C * fC1;
    float pSH18 = fTmp1C * fS1;
    float pSH23 = fTmp2B * fC2;
    float pSH17 = fTmp2B * fS2;
    float pSH24 = 0.6258357354491763f * fC3;
    float pSH16 = 0.6258357354491763f * fS3;
    gpuAtomicAdd(&v_coeffs[16], (scalar_t)(v_sh_sum * pSH16));
    gpuAtomicAdd(&v_coeffs[17], (scalar_t)(v_sh_sum * pSH17));
    gpuAtomicAdd(&v_coeffs[18], (scalar_t)(v_sh_sum * pSH18));
    gpuAtomicAdd(&v_coeffs[19], (scalar_t)(v_sh_sum * pSH19));
    gpuAtomicAdd(&v_coeffs[20], (scalar_t)(v_sh_sum * pSH20));
    gpuAtomicAdd(&v_coeffs[21], (scalar_t)(v_sh_sum * pSH21));
    gpuAtomicAdd(&v_coeffs[22], (scalar_t)(v_sh_sum * pSH22));
    gpuAtomicAdd(&v_coeffs[23], (scalar_t)(v_sh_sum * pSH23));
    gpuAtomicAdd(&v_coeffs[24], (scalar_t)(v_sh_sum * pSH24));

    float fTmp0D_z, fTmp1C_z, fTmp2B_z, fC3_x, fC3_y, fS3_x, fS3_y, pSH20_z,
        pSH21_x, pSH21_z, pSH19_y, pSH19_z, pSH22_x, pSH22_y, pSH22_z, pSH18_x,
        pSH18_y, pSH18_z, pSH23_x, pSH23_y, pSH23_z, pSH17_x, pSH17_y, pSH17_z,
        pSH24_x, pSH24_y, pSH16_x, pSH16_y;
    if (v_dir != nullptr) {
        fTmp0D_z = 3.f * -4.683325804901025f * z2 + 2.007139630671868f;
        fTmp1C_z = 2.f * 3.31161143515146f * z;
        fTmp2B_z = -1.770130769779931f;
        fC3_x = fC2 + x * fC2_x - y * fS2_x;
        fC3_y = x * fC2_y - fS2 - y * fS2_y;
        fS3_x = fS2 + y * fC2_x + x * fS2_x;
        fS3_y = x * fS2_y + fC2 + y * fC2_y;
        pSH20_z = 1.984313483298443f * (pSH12 + z * pSH12_z) +
                  -1.006230589874905f * pSH6_z;
        pSH21_x = fTmp0D;
        pSH21_z = fTmp0D_z * x;
        pSH19_y = fTmp0D;
        pSH19_z = fTmp0D_z * y;
        pSH22_x = fTmp1C * fC1_x;
        pSH22_y = fTmp1C * fC1_y;
        pSH22_z = fTmp1C_z * fC1;
        pSH18_x = fTmp1C * fS1_x;
        pSH18_y = fTmp1C * fS1_y;
        pSH18_z = fTmp1C_z * fS1;
        pSH23_x = fTmp2B * fC2_x;
        pSH23_y = fTmp2B * fC2_y;
        pSH23_z = fTmp2B_z * fC2;
        pSH17_x = fTmp2B * fS2_x;
        pSH17_y = fTmp2B * fS2_y;
        pSH17_z = fTmp2B_z * fS2;
        pSH24_x = 0.6258357354491763f * fC3_x;
        pSH24_y = 0.6258357354491763f * fC3_y;
        pSH16_x = 0.6258357354491763f * fS3_x;
        pSH16_y = 0.6258357354491763f * fS3_y;

        v_x += v_sh_sum *
               (pSH16_x * coeffs[16] + pSH24_x * coeffs[24] +
                pSH17_x * coeffs[17] + pSH23_x * coeffs[23] +
                pSH18_x * coeffs[18] + pSH22_x * coeffs[22] +
                pSH21_x * coeffs[21]);
        v_y += v_sh_sum *
               (pSH16_y * coeffs[16] + pSH24_y * coeffs[24] +
                pSH17_y * coeffs[17] + pSH23_y * coeffs[23] +
                pSH18_y * coeffs[18] + pSH22_y * coeffs[22] +
                pSH19_y * coeffs[19]);
        v_z += v_sh_sum *
               (pSH20_z * coeffs[20] + pSH21_z * coeffs[21] +
                pSH19_z * coeffs[19] + pSH22_z * coeffs[22] +
                pSH18_z * coeffs[18] + pSH23_z * coeffs[23] +
                pSH17_z * coeffs[17]);

        if (v_dir != nullptr) {
            v_dir->x = v_x;
            v_dir->y = v_y;
            v_dir->z = v_z;
        }
    }
}

} // namespace gsplat