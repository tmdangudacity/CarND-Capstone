#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import math

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.waypoints_2d = None
        self.waypoint_tree = None

        #rospy.spin()
        self.simulator_loop_test()

    def simulator_loop_test(self):

        #Testing simulation vehicle traffic data at 10Hz
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            light_wp, state = self.process_traffic_lights()

            if self.state != state:
                self.state_count = 0
                self.state = state

            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                self.last_wp = light_wp if (state == TrafficLight.RED or state == TrafficLight.YELLOW) else -1

            self.state_count += 1

            self.upcoming_red_light_pub.publish(Int32(self.last_wp))

            #rospy.logwarn("Published Closest StopLightWpId {0}".format(self.last_wp))

            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):

        self.waypoints = waypoints

        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):

        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state

        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            self.last_wp = light_wp if(state == TrafficLight.RED or state == TrafficLight.YELLOW) else -1

        self.state_count += 1

        self.upcoming_red_light_pub.publish(Int32(self.last_wp))

        #rospy.logwarn("Published Closest StopLightWpId: {0}".format(self.last_wp))

    def get_closest_waypoint_idx(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        return closest_idx

    def get_light_state(self, light):

        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        """
        # If running with real image
        if(not self.has_image):
            self.prev_light_loc = None
            return TrafficLight.UNKNOWN

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)
        tl_img_state = self.light_classifier.get_classification(cv_image)
        rospy.logwarn("Traffic light image state: {0}".format(tl_img_state))
        """

        #Running with simulation, simply return the light.state from /vehicle/traffic_light
        return light.state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        min_d_idx = None
        line_wp_idx = -1
        light_state = TrafficLight.UNKNOWN

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']


        if(self.pose and self.waypoints and self.waypoint_tree):

            car_wp_idx = self.get_closest_waypoint_idx(self.pose.pose.position.x, self.pose.pose.position.y)
            min_d_idx = len(self.waypoints.waypoints)

            #Find the closest visible traffic light if one exists
            for (i, light) in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint_idx(line[0], line[1])

                # Find the nearest pair of light/stop_line
                d_idx = temp_wp_idx - car_wp_idx

                #rospy.logwarn(" - Waypoints: {0}, StopLines: {1}, CarWpId {2}, StopLineWpId {3}, D_Idx: {4}".format(
                #                    len(self.waypoints.waypoints),
                #                    len(stop_line_positions),
                #                    car_wp_idx,
                #                    temp_wp_idx,
                #                    d_idx))

                if(d_idx >= 0 and d_idx < min_d_idx):
                    min_d_idx = d_idx
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        if closest_light:
            light_state = self.get_light_state(closest_light)

        #rospy.logwarn("Closest StopLineWpId: {0}, Min_D_Idx: {1}, State: {2}".format(line_wp_idx, min_d_idx, light_state))

        return line_wp_idx, light_state

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
