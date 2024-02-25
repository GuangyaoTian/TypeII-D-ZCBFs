#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from visualization_msgs.msg import Marker
import re


marker_pub = None
last_marker_ids = set()
def obstacle_info_callback(data):
    global last_marker_ids
    obstacle_info = data.data
    new_marker_ids = set()

    formatted_list = extract_and_format_data(obstacle_info)
    draw_circles(formatted_list,new_marker_ids)

    for marker_id in last_marker_ids - new_marker_ids:
        delete_marker(marker_id)
    
    last_marker_ids = new_marker_ids

   

def extract_and_format_data(text):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    formatted_data = []
    for i in range(0, len(numbers), 5):
        obstacle_data = [round(float(numbers[i]), 2), round(float(numbers[i+1]), 2),
                         round(float(numbers[i+2]), 2), round(float(numbers[i+3]), 2),
                         round(float(numbers[i+4]), 2)]
        formatted_data.append(obstacle_data)
    return formatted_data

def draw_circles(formatted_data, new_marker_ids):
    global marker_pub
    marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    rospy.sleep(0.1)  
    for i, obstacle in enumerate(formatted_data):
        marker = Marker()
        marker.header.frame_id = "odom"  
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = obstacle[2] * 2  
        marker.color.a = 1.0  
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = obstacle[0]
        marker.pose.position.y = obstacle[1]
        marker.pose.position.z = 0

        marker.id=i

        if obstacle[3] == 0 and obstacle[4] == 0:  
            marker.color.g = 1.0
            new_marker_ids.add(i)
            marker_pub.publish(marker)
        else:  
            marker.color.r = 1.0
            marker_pub.publish(marker)  

            new_marker_ids.add(i)

            
            marker.pose.position.x += obstacle[3] * 0.3
            marker.pose.position.y += obstacle[4] * 0.3
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
            marker.color.a = 0.5  
            marker.scale.x = marker.scale.y = marker.scale.z = obstacle[2] * 2  
            marker.pose.orientation.w = 1.0
            marker.id = i + 10 
            marker.type = marker.SPHERE
            marker.action = marker.ADD

            new_marker_ids.add(i+10)
            marker_pub.publish(marker)

            marker.pose.position.x += obstacle[3] * 0.3
            marker.pose.position.y += obstacle[4] * 0.3
            marker.id = i + 20 
            marker.action = marker.ADD
            new_marker_ids.add(i+20)
            marker_pub.publish(marker)

            marker.pose.position.x += obstacle[3] * 0.3
            marker.pose.position.y += obstacle[4] * 0.3
            marker.id = i + 30 
            marker.action = marker.ADD
            new_marker_ids.add(i+30)
            marker_pub.publish(marker)



        
        ###marker_pub.publish(marker)

def delete_marker(marker_id):
    global marker_pub
    marker = Marker()
    marker.header.frame_id = "base_scan"
    marker.type = Marker.SPHERE
    marker.action = Marker.DELETE
    marker.id = marker_id
    marker_pub.publish(marker)

def listener():
    global marker_pub
    rospy.init_node('obstacle_visualizer', anonymous=True)
    marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    rospy.Subscriber("/obstacle_info", String, obstacle_info_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()

