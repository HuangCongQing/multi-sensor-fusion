// Generated by gencpp from file lidar_imu_fusion/reset_lidar_odometryRequest.msg
// DO NOT EDIT!


#ifndef LIDAR_IMU_FUSION_MESSAGE_RESET_LIDAR_ODOMETRYREQUEST_H
#define LIDAR_IMU_FUSION_MESSAGE_RESET_LIDAR_ODOMETRYREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace lidar_imu_fusion
{
template <class ContainerAllocator>
struct reset_lidar_odometryRequest_
{
  typedef reset_lidar_odometryRequest_<ContainerAllocator> Type;

  reset_lidar_odometryRequest_()
    : resetCloud(false)  {
    }
  reset_lidar_odometryRequest_(const ContainerAllocator& _alloc)
    : resetCloud(false)  {
  (void)_alloc;
    }



   typedef uint8_t _resetCloud_type;
  _resetCloud_type resetCloud;





  typedef boost::shared_ptr< ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator> const> ConstPtr;

}; // struct reset_lidar_odometryRequest_

typedef ::lidar_imu_fusion::reset_lidar_odometryRequest_<std::allocator<void> > reset_lidar_odometryRequest;

typedef boost::shared_ptr< ::lidar_imu_fusion::reset_lidar_odometryRequest > reset_lidar_odometryRequestPtr;
typedef boost::shared_ptr< ::lidar_imu_fusion::reset_lidar_odometryRequest const> reset_lidar_odometryRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator1> & lhs, const ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator2> & rhs)
{
  return lhs.resetCloud == rhs.resetCloud;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator1> & lhs, const ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace lidar_imu_fusion

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "97c883eac274b5605a8b0fd571ab9bc6";
  }

  static const char* value(const ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x97c883eac274b560ULL;
  static const uint64_t static_value2 = 0x5a8b0fd571ab9bc6ULL;
};

template<class ContainerAllocator>
struct DataType< ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "lidar_imu_fusion/reset_lidar_odometryRequest";
  }

  static const char* value(const ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "bool resetCloud\n"
;
  }

  static const char* value(const ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.resetCloud);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct reset_lidar_odometryRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::lidar_imu_fusion::reset_lidar_odometryRequest_<ContainerAllocator>& v)
  {
    s << indent << "resetCloud: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.resetCloud);
  }
};

} // namespace message_operations
} // namespace ros

#endif // LIDAR_IMU_FUSION_MESSAGE_RESET_LIDAR_ODOMETRYREQUEST_H
