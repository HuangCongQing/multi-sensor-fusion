// Generated by gencpp from file lidar_imu_fusion/reset_lidar_odometry.msg
// DO NOT EDIT!


#ifndef LIDAR_IMU_FUSION_MESSAGE_RESET_LIDAR_ODOMETRY_H
#define LIDAR_IMU_FUSION_MESSAGE_RESET_LIDAR_ODOMETRY_H

#include <ros/service_traits.h>


#include <lidar_imu_fusion/reset_lidar_odometryRequest.h>
#include <lidar_imu_fusion/reset_lidar_odometryResponse.h>


namespace lidar_imu_fusion
{

struct reset_lidar_odometry
{

typedef reset_lidar_odometryRequest Request;
typedef reset_lidar_odometryResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct reset_lidar_odometry
} // namespace lidar_imu_fusion


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::lidar_imu_fusion::reset_lidar_odometry > {
  static const char* value()
  {
    return "56eadbef617692a84ca956889d4c0793";
  }

  static const char* value(const ::lidar_imu_fusion::reset_lidar_odometry&) { return value(); }
};

template<>
struct DataType< ::lidar_imu_fusion::reset_lidar_odometry > {
  static const char* value()
  {
    return "lidar_imu_fusion/reset_lidar_odometry";
  }

  static const char* value(const ::lidar_imu_fusion::reset_lidar_odometry&) { return value(); }
};


// service_traits::MD5Sum< ::lidar_imu_fusion::reset_lidar_odometryRequest> should match
// service_traits::MD5Sum< ::lidar_imu_fusion::reset_lidar_odometry >
template<>
struct MD5Sum< ::lidar_imu_fusion::reset_lidar_odometryRequest>
{
  static const char* value()
  {
    return MD5Sum< ::lidar_imu_fusion::reset_lidar_odometry >::value();
  }
  static const char* value(const ::lidar_imu_fusion::reset_lidar_odometryRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::lidar_imu_fusion::reset_lidar_odometryRequest> should match
// service_traits::DataType< ::lidar_imu_fusion::reset_lidar_odometry >
template<>
struct DataType< ::lidar_imu_fusion::reset_lidar_odometryRequest>
{
  static const char* value()
  {
    return DataType< ::lidar_imu_fusion::reset_lidar_odometry >::value();
  }
  static const char* value(const ::lidar_imu_fusion::reset_lidar_odometryRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::lidar_imu_fusion::reset_lidar_odometryResponse> should match
// service_traits::MD5Sum< ::lidar_imu_fusion::reset_lidar_odometry >
template<>
struct MD5Sum< ::lidar_imu_fusion::reset_lidar_odometryResponse>
{
  static const char* value()
  {
    return MD5Sum< ::lidar_imu_fusion::reset_lidar_odometry >::value();
  }
  static const char* value(const ::lidar_imu_fusion::reset_lidar_odometryResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::lidar_imu_fusion::reset_lidar_odometryResponse> should match
// service_traits::DataType< ::lidar_imu_fusion::reset_lidar_odometry >
template<>
struct DataType< ::lidar_imu_fusion::reset_lidar_odometryResponse>
{
  static const char* value()
  {
    return DataType< ::lidar_imu_fusion::reset_lidar_odometry >::value();
  }
  static const char* value(const ::lidar_imu_fusion::reset_lidar_odometryResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // LIDAR_IMU_FUSION_MESSAGE_RESET_LIDAR_ODOMETRY_H
