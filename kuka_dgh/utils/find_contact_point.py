'''
Script to determine manually the contact point of KUKA arm in gepetto viewer
The point is placed by the user in gview WORLD frame and the script
returns the translation of that point w.r.t. specified parent joint

Usage: trial and error. Try to place the contact point as desired in gview
Then write down its location in parent joint frame to add manually a frame of interest
'''

import pinocchio as pin
import numpy as np

VISUALIZE = False
VERBOSE   = True

def display_frame(gui, m, name, size=0.02, color=[1,0,0,1]):
    '''
    Display frame axes in gepetto-viewer
        gui   : gepetto visualizer GUI object
        m     : placement of the frame (global coordinates)
        name  : name of the frame
        size  : radius of the frame origin (sphere)
        color : color of the frame origin (sphere)
    '''
    if(gui.nodeExists('world/'+name)):
        gui.deleteNode('world/'+name, True)
    tf = list(pin.SE3ToXYZQUAT(m))
    gui.addSphere('world/'+name, size, color)
    gui.addLandmark('world/'+name, 0.25)
    gui.applyConfiguration('world/'+name, tf)


def compute_sensor_frame_transform(robot, mount_piece_type, viz=VISUALIZE):

    # Load iiwa model
    visual_model = robot.visual_model
    collision_model = robot.collision_model

    # Get the name of the piece to which the CAD origin is attached 
    # This will enable to compute the sensor frame placement w.r.t. its parent joint
    try: 
        assert(mount_piece_type in ['shell', 'ball'])
    except:
        ValueError('Calibration argument mount_piece_type must be in ["shell", "ball"]')
    if(mount_piece_type == 'shell'):
        cad_origin_name = 'assembled_ee'
    elif(mount_piece_type == 'ball'):
        cad_origin_name = 'kuka_to_sensor_mount'
        
    # Frame id of the CAD frame (base of the mount piece)
    cadFrameId = robot.model.getFrameId(cad_origin_name)
    assert(robot.model.frames[cadFrameId].name == cad_origin_name) 
    
    # Frame id of the soft ball at the tip of the mount piece
    ballFrameId = robot.model.getFrameId('contact')
    assert(robot.model.frames[ballFrameId].name == 'contact')
    
    # Sanity check that they both have the same parent 
    # (note that the joints 'EE' and 'ee_to_contact_joint' in xacro are fixed joints
    # also A7 might be fixed if we reduced the model..) 
    assert(robot.model.frames[ballFrameId].parent == robot.model.frames[cadFrameId].parent)
    print("Common parent to CAD and Ball frames = ", robot.model.names[robot.model.frames[ballFrameId].parent])
    
    A6id = robot.model.frames[cadFrameId].parent
    assert(robot.model.names[A6id] == 'A6')
    
    # Careful : hard-coded 'parent' A7 which might not be an actual joint 
    # if the model is reduced to A1-A6 (A7 is then a fixed joint) :
    # e.g. robot.model.frames[cadFrameId].parent returns A6 ---> invalidates calibration
    # So we use the FRAME id of A7 here to avoid confusion  
    parentId = robot.model.getFrameId('A7')
    # assert(robot.model.frames[ballFrameId].parent == parentId)
    assert(robot.model.frames[parentId].name == 'A7') 
    assert(robot.model.frames[parentId].parent == A6id)
    print("Parent of A7 frame = ", robot.model.names[robot.model.frames[parentId].parent])
    A6MA7 = robot.model.frames[parentId].placement
    
    # We compute now A7Ms = A7Mcad * cadMs    in order to get cMs = cMj * A7Ms
    # A7Mcad is determined by hand from CAD spec and URDF
    # cadMs is determined by hand from manufacturer's drawing & cad specs (careful about mounting !!!)
    # cMj is obtained from URDF model 
    
    # Placement of the CAD origin (mount piece) w.r.t. its parent joint (A7)    = 7M6 * 6Mcad
    A7Mcad  = A6MA7.actInv(robot.model.frames[cadFrameId].placement)
    print("Placement of CAD frame w.r.t. A7 = \n", A7Mcad)
    # Placement of the soft ball frame w.r.t. its parent joint (A7)
    A7Mball = A6MA7.actInv(robot.model.frames[ballFrameId].placement)
    print("Placement of BALL frame w.r.t. A7 = \n", A7Mball)

    # Placement of the sensor frame (cf. manufacturer drawing) w.r.t. CAD origin (mount piece) - measured in the CAD software with Qiushi
        # Rot(x, pi/2) followed by Rot(z, -pi/3) 
        # Offset of 15.7mm along original y 
    # THETA = np.pi/6+np.pi/2
    THETA = np.pi/6 - (1/3 - 0.3)*np.pi + np.pi/2  # from measerement on robot.
    tmp_ty =  pin.SE3( np.eye(3), 
                    np.array([0., 0.0157, 0.]) )
    tmp_rx = pin.SE3( pin.rpy.rpyToMatrix(np.pi/2, 0., 0.) , 
                    np.zeros(3) )
    tmp_rz = pin.SE3( pin.rpy.rpyToMatrix(0., 0., THETA) , 
                    np.zeros(3) )
    cadMs = tmp_ty.act(tmp_rx.act(tmp_rz)) 

    # CALIB to make sensor axis & location match the mounting (measured in gepetto-viewer)
    tmp_rot = pin.SE3( pin.utils.rpyToMatrix(0, 0, np.pi), np.zeros(3))
    cadMs = tmp_rot.act(cadMs)
    tmp_trans = pin.SE3(np.eye(3), np.array([0., 0.025, 0])) # match visually but not measured 6.5 cm between end of cone and base of sensor
    # tmp_trans = pin.SE3(np.eye(3), np.array([0., 0.02, 0])) # to match the 6.5 cm distance between end of cone and base of sensor
    cadMs = tmp_trans.act(cadMs)
    
    # Placement of sensor frame w.r.t. parent joint frame 
    A7Ms = A7Mcad.act(cadMs)

    # sensor frame placement w.r.t. its parent joint A6
    A6Ms = A6MA7.act(A7Ms)
        
    # sensor frame placement w.r.t. to contact frame
    cMs = A7Mball.actInv(A7Ms)

    
    # Now we can add the sensor frame to the model 
    sensorFrameId = robot.model.addFrame(pin.Frame("ft_sensor", A6id, cadFrameId, A6Ms, pin.FrameType.OP_FRAME))
    robot.data = robot.model.createData()
    # Check 
    assert(robot.model.frames[sensorFrameId].parent == robot.model.frames[parentId].parent)
    assert(np.linalg.norm(robot.model.frames[sensorFrameId].placement.rotation - A6Ms.rotation) <= 1e-6)
    assert(np.linalg.norm(robot.model.frames[sensorFrameId].placement.translation - A6Ms.translation) <= 1e-6)
    if(VERBOSE):
        print("CAD origin frame    : ", cad_origin_name + "( id = "+str(cadFrameId) + ") ")
        print("Ball contact frame  : ", "contact" + "( id = "+str(ballFrameId) + ") ")
        print("Sensor frame id     : ", "ft_sensor" + "( id = "+str(sensorFrameId) + ") ")
        print("A7 frame id         : ", robot.model.frames[parentId].name + "( id = "+str(parentId) + ") ")
        print("Common parent       : ", robot.model.names[robot.model.frames[sensorFrameId].parent] + "( id = "+str(robot.model.frames[sensorFrameId].parent) + ") ")
        print(" - - - - -  - - - - - - ")
        print("Placement of the FT sensor frame relative to its parent joint A7: \n", robot.model.frames[sensorFrameId].placement)
        print("Placement of the FT sensor frame relative to the contact frame (Center of soft ball) \n", cMs)

    if(viz):
        # Visualize all the frames to make sure everything makes sense
        q0 = np.zeros(robot.model.nq) 
        q0[1] =0.34
        q0[3] = -0.75
        pin.framesForwardKinematics(robot.model, robot.data, q0)

        # Add a small sphere depicting the collision point
        # Placement of sensor frame w.r.t. contact frame 
        # R = np.eye(3)
        # cMs = pin.SE3(R, np.array([0.8, 0.,1.]))

        # Create the display
        viz = pin.visualize.GepettoVisualizer(robot.model, collision_model, visual_model)
        viz.initViewer()
        viz.loadViewerModel()

        viewer      = viz.viewer
        gui         = viewer.gui
        ref_color   = [1., 0., 0., 1.]
        real_color  = [0., 0., 1., 0.3]
        ref_size    = 0.03
        real_size   = 0.02
                

        dt = 1e-2
        viz.display(q0)

        # Get initial EE placement + tf
        oMcad  = robot.data.oMf[cadFrameId]
        oMball = robot.data.oMf[ballFrameId]
        oMs    = robot.data.oMf[sensorFrameId]
        oMj    = robot.data.oMf[parentId]

        display_frame(gui, oMcad, 'cadFrame', color=[0., 0., 0., 1.], size=0.005)
        display_frame(gui, oMball, 'ballFrame', color=[1., 1., 1., 1.], size=0.005)
        display_frame(gui, oMj, 'parentFrame', color=[0.5, 0.4, 0.5, 1.], size=0.005)
        display_frame(gui, oMs, 'sensorFrame', color=[0., 1., 0., 1.], size=0.01)
        # display_frame(gui, cMs, 'contactpoint', color=[0., 1., 1., 1.], size=0.01)

        gui.refresh()

        # Refresh and wait
        viewer.gui.refresh()
        
    return None, A7Ms, cMs, None #, sensorFrameId
    # # The manufacturer's frame convention is that +z points downwards (i.e. inward of arm)
    # # but the way it is currently mounted on the robot makes +z outward of the robot arm 
    # # The compression should give positive force in z but sensor readings are somehow flipped : pressing results in neg z and pulling in positive z

