#!/usr/bin/env python  
import rospy
import tf
from geometry_msgs.msg import TransformStamped

if __name__ == '__main__':
    rospy.init_node('my_tf_broadcaster')
    
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10.0)
    
    while not rospy.is_shutdown():
        br.sendTransform((3, 3, 0),
                         (0, 0, 0, 1),
                         rospy.Time.now(),
                         "my_frame",
                         "odom")
        rate.sleep()
