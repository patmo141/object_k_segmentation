'''
Created on Mar 7, 2015

@author: Patrick
'''
bl_info = {
    "name": "Curvature Features",
    "author": "Patrick Moore",
    "version": (1, 0),
    "blender": (2, 74, 0),
    "location": "",
    "description": "Draws things on object in View3D",
    "warning": "",
    "wiki_url": "", 
    "category": "Object"}


import bpy
import bgl
import blf
import bmesh
import math
from mathutils import Matrix, Vector
import numpy as np 
from bpy_extras import view3d_utils
from bpy.props import BoolProperty, FloatProperty, IntProperty
from bpy_extras.view3d_utils import location_3d_to_region_2d, region_2d_to_vector_3d, region_2d_to_location_3d, region_2d_to_origin_3d



class CuspWaterDroplet(object):
    def __init__(self, bmvert, pln_pt, pln_no, curv_id):
        
        self.up_vert = bmvert
        self.dn_vert = bmvert
        self.ind_path = [bmvert.index]
        self.curv_id = curv_ud
        self.settled = False
        self.peaked = False
        self.upH = self.hfunc(bmvert)
        self.dnH = self.hfunc(bmvert)
        
        
        self.alpha = 0.5  #good values 0.4 to 0.6
        
    def hfunc(self,bvert):
        
        #possible to cache hfunc over whole mesh?
        vz = pln_no.dot(v.co - pln_pt)
        K = bmvert.layers.float[self.curv_id]
        H = (1 - self.alpha) * K + self.alpha * vz
        
        return H
    def roll_downhill(self):
        
        vs = vert_neighbors(self.dn_vert)
        Hs = [self.hfunc(v) for v in vs]
        
        if self.H < min(Hs):
            self.settled = True
            
        else:
            self.dn_vert = vs[Hs.index(min(Hs))]
            self.ind_path.append(self.vert.index)
            
    def roll_uphill(self):
        
        vs = vert_neighbors(self.up_vert)
        Hs = [self.hfunc(v) for v in vs]
        
        if self.upH > max(Hs):
            self.peaked = True
            
        else:
            self.up_vert = vs[Hs.index(max(Hs))]
            self.ind_path.insert(0, self.vert.index)        
            
def walk_from_vert(vert, prev_vert, steps, scalar_id, dir = None):
    '''
    will take steps away from vertex and test scalar
    sum over path.
    
    #TODO direcionality?
    #TODO normalize by path length
    #TODO normalize by average scalar value per vertex?
    '''
    #assume they are connected
    if not dir:
        dir = vert.co - prev_vert.co
        dir.normalize()
    
    v_paths = []
    ring = [v for v in vert_neighbors(vert) if v.index != prev_vert.index]
    for v in ring:
        v_paths.append([vert, v])

    for i in range(steps):
        new_paths = []
        for n, vpath in enumerate(v_paths):
            #print('%i iteration and %i path' % (i,n))
            #print(vpath)
            #find the tip of the path and all it's branches
            branches = [v for v in vert_neighbors(vpath[-1]) if v.index != vpath[-2].index]
            if len(branches):
                for b in branches:
                    np = vpath.copy()
                    np.append(b)
                    new_paths.append(np)
            else:
                new_paths.append(path)
                    
        v_paths = new_paths
            
            
        
    
    qualities = [] 
    for path in v_paths:
        quality = 0
        for v in path:
            quality += v[scalar_id]
    
    
        quality *= 1/len(path)
        
        pdir = path[-1].co - path[0].co
        pdir.normalize()
        
        quality *= pdir.dot(dir)
        
        qualities.append(quality)
    
    #print(max(qualities))   
    best = qualities.index(max(qualities))
    
    return v_paths[best]

def walk_down_path (path, steps, scalar_id, dir = None):
    '''
    will take steps away from vertex and test scalar
    sum over path.
    
    #TODO direcionality?
    #TODO normalize by path length
    #TODO normalize by average scalar value per vertex?
    '''
    inds = set([v.index for v in path])
    #assume they are connected
    if not dir:
        dir = path[-1].co - path[-4].co
        dir.normalize()
    
    v_paths = []
    ring = [v for v in vert_neighbors(path[-1]) if v.index not in inds]
    for v in ring:
        v_paths.append([path[-1], v])

    for i in range(steps):
        new_paths = []
        for n, vpath in enumerate(v_paths):
            #print('%i iteration and %i path' % (i,n))
            #print(vpath)
            #find the tip of the path and all it's branches, and terminate any which loop back onto original path
            branches = [v for v in vert_neighbors(vpath[-1]) if v.index != vpath[-2].index and v.index not in inds]
            if len(branches):
                for b in branches:
                    np = vpath.copy()
                    np.append(b)
                    new_paths.append(np)
            else:
                new_paths.append(path)
                    
        v_paths = new_paths
            
            
        
    
    qualities = [] 
    for path in v_paths:
        quality = 0
        for v in path:
            quality += v[scalar_id]
    
    
        quality *= 1/len(path)
        
        pdir = path[-1].co - path[0].co
        pdir.normalize()
        
        quality *= pdir.dot(dir)
        
        qualities.append(quality)
    
    #print(max(qualities))   
    best = qualities.index(max(qualities))
    
    return v_paths[best]

def vert_neighbors(bmv):
    '''
    todo, locations, indices or actual verts.
    reverse?
    '''
    verts = [ed.other_vert(bmv) for ed in bmv.link_edges]
    
    return verts
           
    
def calc_curvature(v):
    if False in [ed.is_manifold for ed in v.link_edges]:
        return 0
    
    if not v.is_manifold:
        return 0
    
    position = v.co
    # get vertex normal as a matrix
    normal = v.normal
    Nvi = np.matrix(normal)
    # get sorted 1-ring
    ring = vert_neighbors(v)
    
    
        
    # calculate face weightings, wij
    wij = []
    n = len(ring)
    for j in range(n):
        vec0 = ring[(j+(n-1))%n].co - position
        vec1 = ring[j].co - position
        vec2 = ring[(j+1)%n].co - position
        # Assumes closed manifold
        # TODO: handle boundaries
        wij.append(0.5 * (vec0.cross(vec1).length + 
                     vec1.cross(vec2).length))
    wijSum = sum(wij)
    # calculate matrix, Mvi
    Mvi = np.matrix(np.zeros((3,3)))
    I = np.matrix(np.identity(3))
    for j in range(n):
        vec = ring[j].co - position
        edgeAsMatrix = np.matrix(vec)
        Tij = edgeAsMatrix * (I - (Nvi * Nvi.transpose()))
        Tij *= 1 / np.linalg.norm(Tij)
        kij = (Nvi.transpose() * 2 * edgeAsMatrix)[0,0] / math.pow(vec.length, 2) #may try transposing this once i get into GIT Rep
        Mvi += np.multiply(Tij * Tij.transpose() , wij[j]/wijSum * kij)
    # get eigenvalues and eigenvectors for Mvi
    evals, evecs = np.linalg.eig(Mvi)
    # replace eigenvector matrix with list of Vector3f
    evecs = [Vector((evecs[0,k], evecs[1,k], evecs[2,k]))for k in range(3)]
    # scale eigenvectors by corresponding eigenvalues
    #[e.Unitize() for e in evecs]  alreadt normalized from numpy
    evecs = [evals[k] * evecs[k] for k in range(3)]
    # sort by absolute value of eigenvalues (norm < min < max)
    # sortv: abs curvature, curvature, Vector3f dir
        
    mags = [abs(ev) for ev in evals]
    max_ind = mags.index(max(mags))
    norm_ind = mags.index(min(mags))
    min_ind = list(set([0,1,2]) - set([max_ind, norm_ind]))[0]
    
    return evals[max_ind]
        

def aniso_smooth(bmv, max_id):
    if bmv[max_id] == 0:
        return 0
    ring = vert_neighbors(bmv)
    curvs = [vert[max_id] for vert in ring]
    
    if bmv[max_id] < 0:
        curvs.sort(reverse = True)
    else:
        curvs.sort()
    
    new_curv = .6 * bmv[max_id] +.3*curvs[0] + .1*curvs[1]     
    
    return new_curv

def curvature_on_mesh(bme):
    '''calc initial curvature on bmesh'''
    
    #create custom data layers
    
    if 'max_curve' in bme.verts.layers.float:
        print('Data Layer Exists')
        max_id = bme.verts.layers.float['max_curve']
    else:
        max_id = bme.verts.layers.float.new('max_curve')
        print('Created Data Layer')
    
    for v in bme.verts:
        v[max_id] = calc_curvature(v)
    
       
def smooth_scalar_mesh_curvature(bme):
    max_id = bme.verts.layers.float['max_curve']
    new_curves = [aniso_smooth(bmv, max_id) for bmv in bme.verts] 
    for v in bme.verts:
        v[max_id] = new_curves[v.index]

    return np.mean(new_curves), np.std(new_curves)
 
def main():
    ob = bpy.context.object
    bme = bmesh.new()
    bme.from_mesh(ob.data)

    
    curvature_on_mesh(bme)
    
    max_id = max_id = bme.verts.layers.float['max_curve']
    
    for i in range(0,5):
        avg, std_dev = smooth_scalar_mesh_curvature(bme)

    curves = [v[max_id] for v in bme.verts]
    avg = np.mean(curves)
    std_dev = np.std(curves)
    for v in bme.verts:
        if v[max_id] > avg + .2 * std_dev:
            v.select = True
    
    bme.to_mesh(ob.data)
    bme.free()

def draw_3d_points(context, points, color, size):
    '''
    draw a bunch of dots
    args:
        points: a list of tuples representing x,y SCREEN coordinate eg [(10,30),(11,31),...]
        color: tuple (r,g,b,a)
        size: integer? maybe a float
    '''
    points_2d = [location_3d_to_region_2d(context.region, context.space_data.region_3d, loc) for loc in points]

    bgl.glColor4f(*color)
    bgl.glPointSize(size)
    bgl.glBegin(bgl.GL_POINTS)
    for coord in points_2d:
        #TODO:  Debug this problem....perhaps loc_3d is returning points off of the screen.
        if coord:
            bgl.glVertex2f(*coord)  
        else:
            print('how the f did nones get in here')
            print(coord)
    
    bgl.glEnd()   
    return




class ViewOperatorObjectCurve(bpy.types.Operator):
    """Modal object selection with a ray cast"""
    bl_idname = "view3d.modal_operator_raycast"
    bl_label = "RayCast View Operator"

    recalc_curvature = BoolProperty(
            name="recalc curvature",
            description = "Recalc Curvature (Slow)",
            default=False,
            )
    
    smooth_curvature = BoolProperty(
            name="smooth curvature",
            description = "Smooth Existing Curvature",
            default=False,
            )
    
    smooth = IntProperty(
            name="Smooth Iters",
            default=1,
            min = 0,
            max = 4,
            )
    def draw_callback_verts(self, context):


    
        mx = context.object.matrix_world
        coords = [v.co for v in self.path]
        draw_3d_points(context, coords, (1,.1,.1,1), 4)
        
     

        
    def modal(self, context, event):
        context.area.tag_redraw()
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            # allow navigation
            return {'PASS_THROUGH'}
        
        
        elif event.type == 'MOUSEMOVE':
            return {'RUNNING_MODAL'}
        
        
        elif event.type == 'G' and event.value == 'PRESS':
            self.start_walking(context, event)
            return {'RUNNING_MODAL'}
        
        elif event.type == 'E' and event.value == 'PRESS':
            self.take_step(context,event)
            return {'RUNNING_MODAL'}
            
        elif event.type == 'LEFTMOUSE':
            self.pick_face(context, event)
            return {'RUNNING_MODAL'}
        
        elif event.type == 'UP_ARROW' and event.value == 'PRESS':
            self.big_steps += 1
            return {'RUNNING_MODAL'}
        elif event.type == 'DOWN_ARROW' and event.value == 'PRESS':
            if self.stpes > 3:
                self.big_steps -= 1
                
            return {'RUNNING_MODAL'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self.bme.to_mesh(context.object.data)
            self.bme.free()
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        self.steps = 4
        self.seed = None
        self.seed1 = None
        self.path = []
        self.path_inds = set()
        
        ob = bpy.context.object
        self.bme = bmesh.new()
        self.bme.from_mesh(ob.data)
        self.bme.verts.ensure_lookup_table()
        self.bme.faces.ensure_lookup_table()
    
        if self.recalc_curvature or 'max_curve' not in self.bme.verts.layers.float: 
            curvature_on_mesh(self.bme)
    
        self.max_id = self.bme.verts.layers.float['max_curve']
    
        if self.smooth_curvature:
            for i in range(0,self.smooth):
                smooth_scalar_mesh_curvature(self.bme)
        
        if context.space_data.type == 'VIEW_3D':
            
            self._handle = bpy.types.SpaceView3D.draw_handler_add(self.draw_callback_verts, (context,), 'WINDOW', 'POST_PIXEL')
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Active space must be a View3d")
            return {'CANCELLED'}

    def pick_face(self, context, event, ray_max=1000.0):
        """Run this function on left mouse, execute the ray cast"""
        # get the context arguments
        scene = context.scene
        region = context.region
        rv3d = context.region_data
        coord = event.mouse_region_x, event.mouse_region_y
        obj = context.object
        
        # get the ray from the viewport and mouse
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    
        if rv3d.view_perspective == 'ORTHO':
            # move ortho origin back
            ray_origin = ray_origin - (view_vector * (ray_max / 2.0))
    
        ray_target = ray_origin + (view_vector * ray_max)
        
        def obj_ray_cast(obj, matrix):
            """Wrapper for ray casting that moves the ray into object space"""
    
            # get the ray relative to the object
            matrix_inv = matrix.inverted()
            ray_origin_obj = matrix_inv * ray_origin
            ray_target_obj = matrix_inv * ray_target
    
            # cast the ray
            hit, normal, face_index = obj.ray_cast(ray_origin_obj, ray_target_obj)
    
            if face_index != -1:
                return hit, normal, face_index
            else:
                return None, None, None
            
        hit, normal, face_index = obj_ray_cast(obj, obj.matrix_world)
        if hit is not None:
            curvs = [v[self.max_id] for v in self.bme.faces[face_index].verts]
            self.seed = self.bme.faces[face_index].verts[curvs.index(max(curvs))]
            
            
    def start_walking(self,context, event):
        if not self.seed:
            return
        # get the context arguments
        scene = context.scene
        region = context.region
        rv3d = context.region_data
        coord = event.mouse_region_x, event.mouse_region_y
        obj = context.object
        
        ray_max = 1000.0
        # get the ray from the viewport and mouse
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    
        if rv3d.view_perspective == 'ORTHO':
            # move ortho origin back
            ray_origin = ray_origin - (view_vector * (ray_max / 2.0))
    
        ray_target = ray_origin + (view_vector * ray_max)
        
        def obj_ray_cast(obj, matrix):
            """Wrapper for ray casting that moves the ray into object space"""
    
            # get the ray relative to the object
            matrix_inv = matrix.inverted()
            ray_origin_obj = matrix_inv * ray_origin
            ray_target_obj = matrix_inv * ray_target
    
            # cast the ray
            hit, normal, face_index = obj.ray_cast(ray_origin_obj, ray_target_obj)
    
            if face_index != -1:
                return hit, normal, face_index
            else:
                return None, None, None
            
        hit, normal, face_index = obj_ray_cast(obj, obj.matrix_world)
        if hit is not None:
            dir = hit - self.seed.co
            dir.normalize()
            
            qualities = []
            neighbors = vert_neighbors(self.seed)
            for v in neighbors:
                qdir = v.co - self.seed.co
                qdir.normalize()
                #qdir *= -1
                q = qdir.dot(dir) *  v[self.max_id]
                qualities.append(q)
                
            
            self.seed1 = neighbors[qualities.index(max(qualities))]
            
            self.path = walk_from_vert(self.seed1, self.seed, self.steps, self.max_id, dir = dir)
            
            for v in self.path:
                self.path_inds.add(v.index)
                v.select = True
            
    def take_step(self,context, event):
        print('take step')
        if not len(self.path):
            return
              
        dir = self.path[-1].co - self.path[-4].co
        dir.normalize()
            
        extension =  walk_down_path(self.path, self.steps, self.max_id, dir = dir)
        print(extension[1:])
        #print("the extension is %i long" % len(extension))
        self.path.extend(extension[1:])
        
        print(len(self.path)) 
        for v in self.path:
            v.select = True


class ViewObjectSalience(bpy.types.Operator):
    """View Object Salience"""
    bl_idname = "view3d.live_preview_salience"
    bl_label = "View Salience"

    
    def draw_callback_salient(self, context):

        # draw some text
        blf.position(0, 15, 30, 0)
        blf.size(0, 20, 72)
        blf.draw(0, "Salient Width " + str(self.width)[0:4])
        
        blf.position(0, 15, 80, 0)
        blf.size(0, 20, 72)
        blf.draw(0, "Island Size " + str(self.island_min)[0:4])
        
        mx = context.object.matrix_world
            
        
        if len(self.islands):
            for island in self.islands:
                coords = [mx * self.bme.verts[i].co for i in island]                        
                draw_3d_points(context, coords, (.1,.5,1,1), 3) 
        else:
            coords = [mx * self.bme.verts[i].co for i in self.salient_verts]
            draw_3d_points(context, coords, (1,.1,.1,1), 2)
            
    def find_salient_islands(self,context):
        
        untested = set(self.salient_verts)
        still_salient = set()
        def get_island(v):
            tested = set()
            tested.add(v)
            new_verts = [v.index for v in vert_neighbors(self.bme.verts[v]) if v.index in untested]
            tested = tested.union(set(new_verts))
            
            n_iters = 0
            while new_verts and n_iters < 400:
                new_neighbors = set()
                for nv in new_verts:
                    ring = set([v.index for v in vert_neighbors(self.bme.verts[nv]) if v.index in untested])
                    new_immediate_neighbors =  ring - tested
                    new_neighbors = new_neighbors.union(new_immediate_neighbors)
                    tested = tested.union(set(new_neighbors))
                
                n_iters += 1    
                new_verts = list(new_neighbors)
                
            if n_iters == 30:
                print("found %i verts but itered out" % len(tested))
            
            return tested 
        
        iters = 0
        while len(untested) and iters < 5000:
            vrt = untested.pop()
            island = get_island(vrt)
            
            untested = untested.difference(island)
            
            if len(island) > self.island_min:
                self.islands.append(list(island))
            
            
            iters += 1
            
        print('%i salient verts remaining' % len(untested))
        
    def update_salience(self,context):
        self.salient_verts = [v.index for v in self.bme.verts if v[self.max_id] > self.avg + self.width * self.std_dev]
        self.islands = []
    def invoke(self,context, event):
        ob = bpy.context.object
        self.bme = bmesh.new()
        self.bme.from_mesh(ob.data)

        self.bme.verts.ensure_lookup_table()
        self.bme.faces.ensure_lookup_table()
        
        self.max_id = self.bme.verts.layers.float['max_curve']
        
        curves = [v[self.max_id] for v in self.bme.verts]
        self.avg = np.mean(curves)
        self.std_dev = np.std(curves)

        self.width = .5
        self.island_min = 80
        self.islands = []
        self.salient_verts = [v.index for v in self.bme.verts if v[self.max_id] > self.avg + self.width * self.std_dev]
        
        self._handle = bpy.types.SpaceView3D.draw_handler_add(self.draw_callback_salient, (context,), 'WINDOW', 'POST_PIXEL')
        context.window_manager.modal_handler_add(self)
    
        return {'RUNNING_MODAL'}
    
    def modal(self,context,event):
        context.area.tag_redraw()
        
        if event.type == 'RET' and event.value == 'PRESS':
            for i in self.salient_verts:
                self.bme.verts[i].select = True
            
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self.bme.to_mesh(context.object.data)
            self.bme.free()
            
            return {'FINISHED'}
          
        elif event.type == 'U' and event.value == 'PRESS':
            self.update_salience(context)
            return {'RUNNING_MODAL'}
        
        elif event.type == 'UP_ARROW' and event.value == 'PRESS':
            if event.shift:
                self.width += .05
            else:
                self.width += .1
            self.update_salience(context)
            return {'RUNNING_MODAL'}    
        elif event.type == 'DOWN_ARROW' and event.value == 'PRESS':
            if event.shift:
                self.width -= .05
            else:
                self.width -= .1
            self.update_salience(context)
            return {'RUNNING_MODAL'}
        
        elif event.type == 'RIGHT_ARROW' and event.value == 'PRESS':
            if event.shift:
                self.island_min += 5
            else:
                self.island_min += 10
            
            
            self.find_salient_islands(context)
            return {'RUNNING_MODAL'}    
        
        elif event.type == 'LEFT_ARROW' and event.value == 'PRESS':
            if event.shift:
                self.island_min -= 5
            else:
                self.island_min -= 10
            self.find_salient_islands(context)
            return {'RUNNING_MODAL'} 
        
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self.bme.to_mesh(context.object.data)
            self.bme.free()
            return {'CANCELLED'}
        else:
            return {'PASS_THROUGH'}
        
def register():
    bpy.utils.register_class(ViewOperatorObjectCurve)
    bpy.utils.register_class(ViewObjectSalience)
def unregister():
    bpy.utils.unregister_class(ViewOperatorObjectCurve)
    bpy.utils.unregister_class(ViewObjectSalience)
    

if __name__ == "__main__":
    register()

