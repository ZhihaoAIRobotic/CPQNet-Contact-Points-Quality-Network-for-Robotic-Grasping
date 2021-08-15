from find_grasp_function import Grasp_finder

import pyrealsense2 as rs  

import urx
robot=urx.Robot("192.168.1.3")

from kinematics import *
import time

def rotation_vector_to_rotation_matrix(xyzRxRyRz):
    x=xyzRxRyRz[0]
    y=xyzRxRyRz[1]
    z=xyzRxRyRz[2]
    rx=xyzRxRyRz[3]
    ry=xyzRxRyRz[4]
    rz=xyzRxRyRz[5]
    theta=math.sqrt(rx**2+ry**2+rz**2)
    rx=rx/theta
    ry=ry/theta
    rz=rz/theta
    I=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    W=np.array([[0.0,-rz,ry],[rz,0.0,-rx],[-ry,rx,0.0]])
    R=I+W*math.sin(theta)+W.dot(W)*(1-math.cos(theta));
    H=[[R[0][0],R[0][1],R[0][2],x],[R[1][0],R[1][1],R[1][2],y],[R[2][0],R[2][1],R[2][2],z],[0.0,0.0,0.0,1.0]]
    return H
def inverse_homomatrix(H):
     #求齐次坐标矩阵的逆
     n=np.array([H[0][0],H[1][0],H[2][0]])
     o=np.array([H[0][1],H[1][1],H[2][1]])
     a=np.array([H[0][2],H[1][2],H[2][2]])
     p=np.array([H[0][3],H[1][3],H[2][3]])
     pxout=-p.dot(n)
     pyout=-p.dot(o)
     pzout=-p.dot(a)
     Hout=[[H[0][0],H[1][0],H[2][0],pxout],[H[0][1],H[1][1],H[2][1],pyout],[H[0][2],H[1][2],H[2][2],pzout],[0.0,0.0,0.0,1.0]]
     return Hout
def control_graspper(value,n):
    if robot is None:
        return
    try:  
        robot.set_tool_digital_out(0,value)
        time.sleep(n)
    except:
        print("控制夹抓失败！")
#循环抓取
while True:
    #movej init point
    joints=[-1.3423002401935022, -1.721588436757223, -1.0202143828021448, -1.961719814931051, 1.533403754234314, 0.34252750873565674]

    acceleration=0.1*20
    velocity=0.05*20
    joints_now=robot.movej([joints[0],joints[1],joints[2],joints[3],joints[4],joints[5]],acc=acceleration,vel=velocity)   
    xyzRxRyRz=robot.getl()
    control_graspper(False,0)

    homo_camera_takephoto_pos=rotation_vector_to_rotation_matrix(xyzRxRyRz)
    homo_camera_takephoto_pos=np.array(homo_camera_takephoto_pos)
    

      
                         
    homo_robot_to_camera=np.array([[0.002341383381115852, -0.9999540612063075, -0.009294805047711852, 0.0414266],
                               [0.9991550054407569, 0.002720715839324025, -0.04101064261875165, -0.0042289],
                               [0.04103404716261793, -0.009190929350940878, 0.9991154757039459, 0.107203319],
                               [0.0, 0.0, 0.0, 1.0]])
                               
                            
    homo=homo_camera_takephoto_pos.dot(homo_robot_to_camera)

    # Setup:
    pipe = rs.pipeline()
    cfg = rs.config()

    profile = pipe.start(cfg)

    # Skip 5 first frames to give the Auto-Exposure time to adjust
    for x in range(5):
      pipe.wait_for_frames()
  
    # Store next frameset for later processing:
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    # Cleanup:
    pipe.stop()
    print("Frames Captured")

    colorized_depth = np.array(depth_frame.get_data())


    finder = Grasp_finder()
    #path = 'cameradata2/test.npy'
    #d_data = np.load(path)
    homo_object_in_camera=finder.find(colorized_depth/1000)

    homo_object_in_robot=homo.dot(homo_object_in_camera)
    #print(homo_object_in_robot)

    tool=np.array([[0.0,0.0,-1.0,0.0],
                   [0.0,1.0,0.0,0.0],
                   [1.0,0.0,0.0,0.3-0.037],
                   [0, 0, 0, 1]])

    Htoolinv=inverse_homomatrix(tool)
    Htoolinv=np.array(Htoolinv)
    homo_target=homo_object_in_robot.dot(Htoolinv)

    qnear=robot.getj()
    joints_target=inv_kin(homo_target,qnear)
    #print(joints_target)
    acceleration=0.1*10
    velocity=0.05*10
    
    joints_now=robot.movej([joints_target[0],joints_target[1],joints_target[2],joints_target[3],joints_target[4],joints_target[5]],acc=acceleration,vel=velocity)
    control_graspper(True,0.1)
    time.sleep(4)
    acceleration=0.1*20
    velocity=0.05*20
    joints_target[2]=joints_target[2]+0.3
    joints_now=robot.movej([joints_target[0],joints_target[1],joints_target[2],joints_target[3],joints_target[4],joints_target[5]],acc=acceleration,vel=velocity)
    joints_put=[-0.7100852171527308, -1.6953290144549769, -1.704646412526266, -1.264317814503805, 1.6252219676971436, 1.134010672569275]
    joints_now=robot.movej([joints_put[0],joints_put[1],joints_put[2],joints_put[3],joints_put[4],joints_put[5]],acc=acceleration,vel=velocity)
    control_graspper(False,1)





