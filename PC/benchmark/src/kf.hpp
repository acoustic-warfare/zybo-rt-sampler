/**
 * @file kf.hpp
 * @author Irreq
 * @brief A simple linear 3D Kalmanfilter for predicting and filtering measurments
 * @version 0.1
 * @date 2023-07-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <Eigen/Dense>
#include <vector>

using namespace Eigen;

/**
 * @brief Linear 3D Kalmanfilter
 * 
 */
class KalmanFilter3D
{
private:
    Matrix<float, 6, 6> A; // State transition matrix
    Matrix<float, 6, 6> Q; // Process noise covariance
    Matrix<float, 3, 6> H; // Measurement matrix
    Matrix<float, 3, 3> R; // Measurement noise covariance
    Matrix<float, 6, 6> P; // Error covariance matrix
    Vector<float, 6> x;    // State vector

public:
    /**
     * @brief Construct a new Kalman Filter 3 D object
     * 
     */
    KalmanFilter3D()
    {
        A << 1, 0, 0, 1, 0, 0,
             0, 1, 0, 0, 1, 0,
             0, 0, 1, 0, 0, 1,
             0, 0, 0, 1, 0, 0,
             0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 1;

        Q << 0.1, 0, 0, 0, 0, 0,
             0, 0.1, 0, 0, 0, 0,
             0, 0, 0.1, 0, 0, 0,
             0, 0, 0, 0.1, 0, 0,
             0, 0, 0, 0, 0.1, 0,
             0, 0, 0, 0, 0, 0.1;

        H << 1, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0;

        R << 0.1, 0, 0,
             0, 0.1, 0,
             0, 0, 0.1;


        P.setIdentity();
        x.setZero();
    }

    /**
     * @brief Update the internal state matrix with the new measurment
     * 
     * @param measurement 
     */
    void update(const Vector3f &measurement)
    {
        // Prediction
        x = A * x;
        P = A * P * A.transpose() + Q;

        // Kalman gain calculation
        Matrix3f S = H * P * H.transpose() + R;
        Matrix<float, 6, 3> K = P * H.transpose() * S.inverse();

        // Update step
        Vector3f y = measurement - H * x;
        x = x + K * y;
        P = (Matrix<float, 6, 6>::Identity() - K * H) * P;
    }

    /**
     * @brief Get the State object
     * 
     * @return Vector3f 
     */
    Vector3f getState() const
    {
        return x.head(3);
    }

    /**
     * @brief predict N times ahead (Warning fails quite fast)
     * 
     * @param N 
     * @return Vector3f 
     */
    Vector3f predict(int N)
    {
        Matrix<float, 6, 6> An = A;
        Vector<float, 6> xn = x;

        // Apply state transition matrix N times
        for (int i = 0; i < N; ++i)
        {
            xn = An * xn;
            An = An * A;
        }

        return xn.head(3);
    }

    /**
     * @brief Wrapper for predict
     * 
     * @param N 
     * @return std::vector<float> 
     */
    std::vector<float> predictf(int N)
    {
        Vector3f p = predict(N);
        return std::vector<float>{p(0), p(1), p(2)};
    }

    /**
     * @brief Wrapper for Get the Statef object
     * 
     * @return std::vector<float> 
     */
    std::vector<float> getStatef() const
    {
        Vector3f s = getState();
        return std::vector<float>{s(0), s(1), s(2)};
    }

    /**
     * @brief Wrappper for update
     * 
     * @param measurment 
     */
    void updatef(const std::vector<float> &measurment)
    {
        Vector3f m(measurment[0], measurment[1], measurment[2]);
        update(m);
    }
};

// int main()
// {
//     KalmanFilter3D kf;

//     std::vector<Vector3f> measurements = {
//         Vector3f(1, 1, 0),
//         Vector3f(2, 2, 0),
//         Vector3f(3, 4, 0),
//         Vector3f(4, 8, 0),
//         Vector3f(5, 16, 0),
//         Vector3f(6, 32, 0),
//         Vector3f(7, 64, 0),
//         Vector3f(8, 128, 0),
//         Vector3f(9, 256, 0),
//         Vector3f(10, 512, 0)};

//     Vector3f vec_(1, 2, 3);

//     std::cout << "Filtered state: " << vec_(0)<< std::endl;

//     // Apply Kalman filter to each measurement
//     for (const auto &measurement : measurements)
//     {
//         std::cout << "Measured state: " << measurement.transpose() << std::endl;
//         kf.update(measurement);
//         Vector3f state = kf.getState();
//         std::cout << "Filtered state: " << state.transpose() << std::endl;

//         int N = 10; // Number of seconds to predict ahead
//         Vector3f predictedState = kf.predict(N);
//         // std::cout << "Predicted state (" << N << " seconds ahead): " << predictedState.x << ", " << predictedState.y << ", " << predictedState.z << std::endl;
//         std::cout << "Predicted state: " << predictedState.transpose() << " (" << N << " seconds ahead): " << std::endl;
//     }

//     return 0;
// }