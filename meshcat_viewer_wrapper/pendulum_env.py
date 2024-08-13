import pinocchio as pin
import numpy as np
import hppfcl as fcl

class NPendulum:
    ''' N pendulum env
    '''
    def __init__(self, N) -> None:
        model = pin.Model()
        geom_model = pin.GeometryModel()
        parent_id = 0

        base_radius = 0.2
        shape_base = fcl.Sphere(base_radius)
        geom_base = pin.GeometryObject("base", 0, shape_base, pin.SE3.Identity())
        geom_base.meshColor = np.array([1.,0.1,0.1,1.])
        geom_model.addGeometryObject(geom_base)

        joint_placement = pin.SE3.Identity()
        body_mass = 1.
        body_radius = 0.1

        joint_length = [1.5, 1, 0.5]

        for k in range(N):
            joint_name = "joint_" + str(k+1)
            joint_id = model.addJoint(parent_id,pin.JointModelRX(),joint_placement,joint_name)
            
            joint_frame_id = model.addJointFrame(joint_id, parent_id)

            body_inertia = pin.Inertia.FromSphere(body_mass,body_radius)
            body_placement = joint_placement.copy()
            body_placement.translation[2] = joint_length[k]
            model.appendBodyToJoint(joint_id,body_inertia,body_placement)

            geom1_name = "ball_" + str(k+1)
            shape1 = fcl.Sphere(body_radius)
            geom1_obj = pin.GeometryObject(geom1_name, joint_id, shape1, body_placement)
            geom1_obj.meshColor = np.ones((4))
            geom_model.addGeometryObject(geom1_obj)

            geom2_name = "bar_" + str(k+1)
            shape2 = fcl.Cylinder(body_radius/4.,body_placement.translation[2])
            shape2_placement = body_placement.copy()
            shape2_placement.translation[2] /= 2.

            geom2_obj = pin.GeometryObject(geom2_name, joint_id, shape2, shape2_placement)
            geom2_obj.meshColor = np.array([0.,0.,0.,1.])
            geom_model.addGeometryObject(geom2_obj)

            parent_id = joint_id
            joint_placement = body_placement.copy()

        model.addBodyFrame(geom1_name, joint_id, body_placement, joint_frame_id)
        self.geom_model = geom_model
        self.model = model
        
        model.lowerPositionLimit.fill(-np.pi)
        model.upperPositionLimit.fill(+np.pi)
        model.effortLimit.fill(50.)
        

# NPendulum()