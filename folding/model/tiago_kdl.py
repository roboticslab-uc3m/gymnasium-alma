import PyKDL as kdl
import kdl_parser_py.urdf as kdlp
import os
import numpy as np

class TiagoKDL:
    def __init__(self):
        self.start_link = "torso_lift_link"
        self.right_end_link = "gripper_right_grasping_frame"
        self.base_link = "base_footprint"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, 'urdf', 'tiago_dual_gripper.urdf')
        self.right_chain, self.q_limits, self.base_to_start_transform, self.end_to_gripper_transform = self._load_chain_from_urdf(path)

        # right solver
        self.right_fk_solver = kdl.ChainFkSolverPos_recursive(self.right_chain)
        self.right_ik_solver_vel = kdl.ChainIkSolverVel_pinv(self.right_chain)
        self.right_il_solver_pos = kdl.ChainIkSolverPos_LMA(self.right_chain)
        self.right_ik_solver = kdl.ChainIkSolverPos_NR_JL(self.right_chain,self.q_limits[0],self.q_limits[1],self.right_fk_solver, self.right_ik_solver_vel, maxiter = 30)

        self.q_r = kdl.JntArray(self.right_chain.getNrOfJoints())

        # self._start()


    def _start(self):
        # save a posible initial poses (not set)
        r = kdl.Rotation.RPY(0,0,0)
        v_r = kdl.Vector(0.5, 0, 0.6)
        pose_r = kdl.Frame(r,v_r)
        self.right_initial_goal = pose_r

        q_tmp_r = kdl.JntArray(self.right_chain.getNrOfJoints())

        # set home position as initial configuration
        right_home = [-1.1, 1.4679, 2.714, 1.7095, -1.5708, 1.37, 0]

        for i in range(len(right_home)):
            q_tmp_r[i] = right_home[i]

        self.q_r = q_tmp_r

        return
    

    def fk(self, q_r:np.array):
        q_r = q_r.tolist()
        pose_r = kdl.Frame()
        pose_base_r =  kdl.Frame()

        q_r_tmp = kdl.JntArray(self.right_chain.getNrOfJoints())
        for i in range(len(q_r)):
            q_r_tmp[i] = q_r[i]
        
        self.right_fk_solver.JntToCart(q_r_tmp, pose_r)

        #transform to base
        pose_base_r = self.base_to_start_transform * pose_r

        #transform to gripper        
        right_pose_base_gripper = pose_base_r * self.end_to_gripper_transform

        right_pose_base_gripper = self.kdl_to_np(right_pose_base_gripper)

        return  right_pose_base_gripper
    

    def ik_vel(self, vel:np.array, q_r:np.array):

        tw = kdl.Twist()
        tw.vel = kdl.Vector(vel[0], vel[1], vel[2])
        tw.rot = kdl.Vector(vel[3], vel[4], vel[5])

        q_r_current = kdl.JntArray(self.right_chain.getNrOfJoints())
        for i in range(self.right_chain.getNrOfJoints()):
            q_r_current[i] = q_r[i]


        q_tmp_r = kdl.JntArray(self.right_chain.getNrOfJoints())
        ik_status_r = self.right_ik_solver_vel.CartToJnt(q_r_current, tw, q_tmp_r)

        if ik_status_r < 0:
            print(ik_status_r)
            q_tmp_r = kdl.JntArray(self.right_chain.getNrOfJoints())
            q_tmp_r = q_r_current

        q_np = np.zeros(self.right_chain.getNrOfJoints())
        for i in range(self.right_chain.getNrOfJoints()):
            q_np[i] =q_r[i] + 0.01* q_tmp_r[i] 


        return q_np
    


    def ik(self, pose:np.array, q_r:np.array):

        pose = self.np_to_kdl(pose)
        goal_pose_start_r = kdl.Frame()
        goal_pose_start_r = self.base_to_start_transform.Inverse() * pose

        goal_pose_start_end_r = kdl.Frame()
        goal_pose_start_end_r = goal_pose_start_r * self.end_to_gripper_transform.Inverse()

        q_tmp_r = kdl.JntArray(self.right_chain.getNrOfJoints())
        q_current = kdl.JntArray(self.right_chain.getNrOfJoints())
        for i in range(self.right_chain.getNrOfJoints()):
            q_current[i] = q_r[i]

        ik_status_r = self.right_il_solver_pos.CartToJnt(q_current, goal_pose_start_end_r, q_tmp_r)

        if ik_status_r < 0:
            print("PUÑETAAAAAAAAAAAAAA")
            q_tmp_r = kdl.JntArray(self.right_chain.getNrOfJoints())
            q_tmp_r = q_current


        q_r = np.zeros(self.right_chain.getNrOfJoints())
        for i in range(self.right_chain.getNrOfJoints()):
            q_r[i] = q_tmp_r[i]

        return q_r

        


    
    def _load_chain_from_urdf(self, urdf_file):
        
        # load urdf file and chain
        with open(urdf_file) as urdf:
            urdf_model = kdlp.urdf.URDF.from_xml_string(urdf.read())
        succes, tree = kdlp.treeFromUrdfModel(urdf_model)

        right_chain = kdl.Chain()
        right_chain = tree.getChain(self.start_link, self.right_end_link)

        print("Loaded right chain with %d segements.", right_chain.getNrOfJoints())        
        
        # create transform to start 
        base_chain = tree.getChain(self.base_link, self.start_link)
        base_to_start_transform = kdl.Frame()
        fk_base_solver = kdl.ChainFkSolverPos_recursive(base_chain)
        base_q = kdl.JntArray(base_chain.getNrOfJoints())
        fk_base_solver.JntToCart(base_q, base_to_start_transform)

        # create transform to gripper center
        rot = kdl.Rotation.RPY(0,0,0)
        #???
        # trans = kdl.Vector(0.05,0,0) # gripper center
        trans = kdl.Vector(0.05,0,0)
        end_to_gripper_transform = kdl.Frame(rot, trans)

        # get joint limits
        joint_limits = {}
        for joint in urdf_model.joints:
            if joint.type != "fixed":
                if joint.limit is not None:
                    joint_limits[joint.name] = (joint.limit.lower, joint.limit.upper)
        right_q_min = kdl.JntArray(right_chain.getNrOfJoints())
        right_q_max = kdl.JntArray(right_chain.getNrOfJoints())

        for i in range(right_chain.getNrOfJoints()):
            joint_name = right_chain.getSegment(i).getJoint().getName()
            if joint_name in joint_limits:
                lower_limit, upper_limit = joint_limits[joint_name]
                right_q_min[i] = lower_limit
                right_q_max[i] = upper_limit

        q_limits = [right_q_min, right_q_max]

        return right_chain, q_limits, base_to_start_transform, end_to_gripper_transform
    

    def np_to_kdl(self, pose):

        rot = kdl.Rotation.Quaternion( pose[3], pose[4],pose[5], pose[6])
        trans = kdl.Vector(pose[0],pose[1],pose[2])
        kdl_pose = kdl.Frame(rot, trans)

        return kdl_pose
    

    def kdl_to_np(self, kdl_pose):
        x, y, z = kdl_pose.p.x(), kdl_pose.p.y(), kdl_pose.p.z()
        qx, qy, qz, qw = kdl_pose.M.GetQuaternion()
        np_pose = np.array([x, y, z, qx, qy, qz, qw])

        return np_pose.copy()
    