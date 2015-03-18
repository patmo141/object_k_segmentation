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
import random
import time
from collections import Counter
from mathutils import Matrix, Vector
import numpy as np 
from bpy_extras import view3d_utils
from bpy.props import BoolProperty, FloatProperty, IntProperty
from bpy_extras.view3d_utils import location_3d_to_region_2d, region_2d_to_vector_3d, region_2d_to_location_3d, region_2d_to_origin_3d

from retopoflow.polystrips_utilities import cubic_bezier_fit_points, cubic_bezier_blend_t

def vector_average(l_vecs):
    v_mean = Vector((0,0,0))
    for vec in l_vecs:
        v_mean += vec
    v_mean *= 1/len(l_vecs)
    
    return v_mean
    
def bbox(bme_verts):
    '''
    takes a lsit of BMverts ora  list of vectors
    '''
    if hasattr(bme_verts[0], 'co'):
        verts = [v.co for v in bme_verts]
    else:
        verts = [v for v in bme_verts]
        
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))

def closest_point(pt, locs):
    ds = [(loc - pt).length for loc in locs]
    
    best = ds.index(min(ds))
    
    return (best, locs[best], ds[best])


def closest_point_in_list(pt, locs):
    '''
    the test point is included in the list
    '''
    ds = [(loc - pt).length for loc in locs if pt != loc]
    
    best = ds.index(min(ds))
    
    return (best, locs[best], ds[best])


def points_within_radius(pt, locs, R):
    '''
    the test point is included in the list
    '''
    dist_inds = {}
    ds = []
    
    for i, loc in enumerate(locs):
        if pt != loc:
            D = (loc -pt).length
            dist_inds[D] = i
            
            if D < R:
                ds.append(D)
                
    ds.sort()
    inds = [dist_inds[D] for D in ds]
    vs = [locs[i] for i in inds]
    return (ds, inds, vs)
    
    
    
    
    
    best = ds.index(min(ds))
    
    return (best, locs[best], ds[best])


def calculate_plane(locs, itermax = 500, debug = False):
    '''
    args: 
    vertex_locs - a list of type Vector
    return:
    normal of best fit plane
    '''
    if debug:
        start = time.time()
        n_verts = len(locs)
    
    # calculating the center of masss
    com = Vector()
    for loc in locs:
        com += loc
    com /= len(locs)
    x, y, z = com
    
    # creating the covariance matrix
    mat = Matrix(((0.0, 0.0, 0.0),
                  (0.0, 0.0, 0.0),
                  (0.0, 0.0, 0.0),
                   ))
    for loc in locs:
        mat[0][0] += (loc[0]-x)**2
        mat[1][0] += (loc[0]-x)*(loc[1]-y)
        mat[2][0] += (loc[0]-x)*(loc[2]-z)
        mat[0][1] += (loc[1]-y)*(loc[0]-x)
        mat[1][1] += (loc[1]-y)**2
        mat[2][1] += (loc[1]-y)*(loc[2]-z)
        mat[0][2] += (loc[2]-z)*(loc[0]-x)
        mat[1][2] += (loc[2]-z)*(loc[1]-y)
        mat[2][2] += (loc[2]-z)**2
    
    # calculating the normal to the plane
    normal = False
    try:
        mat.invert()
    except:
        if sum(mat[0]) == 0.0:
            normal = Vector((1.0, 0.0, 0.0))
        elif sum(mat[1]) == 0.0:
            normal = Vector((0.0, 1.0, 0.0))
        elif sum(mat[2]) == 0.0:
            normal = Vector((0.0, 0.0, 1.0))
    if not normal:
        # warning! this is different from .normalize()
        iters = 0
        vec = Vector((1.0, 1.0, 1.0))
        vec2 = (mat * vec)/(mat * vec).length
        while vec != vec2 and iters < itermax:
            iters+=1
            vec = vec2
            vec2 = mat * vec
            if vec2.length != 0:
                vec2 /= vec2.length
        if vec2.length == 0:
            vec2 = Vector((1.0, 1.0, 1.0))
        normal = vec2


    if debug:
        if iters == itermax:
            print("looks like we maxed out our iterations")
        print("found plane normal for %d verts in %f seconds" % (n_verts, time.time() - start))
    
    return com, normal


class CuspWaterDroplet(object):
    def __init__(self, bmvert, pln_pt, pln_no, curv_id):
        
        self.up_vert = bmvert
        self.dn_vert = bmvert
        self.ind_path = [bmvert.index]
        self.curv_id = curv_id
        self.settled = False
        self.peaked = False
        
        
        self.pln_no = pln_no
        self.pln_pt = pln_pt
        self.alpha = 0.5  #good values 0.4 to 0.6
        
        self.upH = self.hfunc(bmvert)
        self.dnH = self.hfunc(bmvert)
        
    def hfunc(self,bvert):
        
        #possible to cache hfunc over whole mesh?
        vz = self.pln_no.dot(bvert.co - self.pln_pt)
        K = bvert[self.curv_id]  #curvatures are precalced and saved in ID layer
        H = (1 - self.alpha) * K - self.alpha * vz  #perhaps height and curvature need to be normalized somehow?
        
        return H
    def roll_downhill(self):
        
        vs = vert_neighbors(self.dn_vert)
        Hs = [self.hfunc(v) for v in vs]
        
        if self.dnH < min(Hs):
            self.settled = True
            return
        
        else:
            V = vs[Hs.index(min(Hs))]
            self.dn_vert = V
            self.ind_path.append(self.dn_vert.index)
            self.dnH = min(Hs)
            
    def roll_uphill(self):
        
        vs = vert_neighbors(self.up_vert)
        Hs = [self.hfunc(v) for v in vs]
        
        if self.upH > max(Hs):
            self.peaked = True
            return
        
        else:
            V = vs[Hs.index(max(Hs))]            
            self.up_vert = V
            self.ind_path.insert(0, self.up_vert.index)        
            self.upH = max(Hs)
            
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

def draw_polyline_from_3dpoints(context, points_3d, color, thickness, LINE_TYPE = "GL_LINE_STIPPLE"):
    '''
    a simple way to draw a line
    slow...becuase it must convert to screen every time
    but allows you to pan and zoom around
    
    args:
        points_3d: a list of tuples representing x,y SCREEN coordinate eg [(10,30),(11,31),...]
        color: tuple (r,g,b,a)
        thickness: integer? maybe a float
        LINE_TYPE:  eg...bgl.GL_LINE_STIPPLE or 
    '''
    
    points = [location_3d_to_region_2d(context.region, context.space_data.region_3d, loc) for loc in points_3d]
    
    #if LINE_TYPE == "GL_LINE_STIPPLE":  
    #    bgl.glLineStipple(4, 0x5555)  #play with this later
    #    bgl.glEnable(bgl.GL_LINE_STIPPLE)  
    bgl.glEnable(bgl.GL_BLEND)
    
    bgl.glColor4f(*color)
    bgl.glLineWidth(thickness)
    bgl.glBegin(bgl.GL_LINE_STRIP)
    for coord in points:
        if coord:
            bgl.glVertex2f(*coord)
    
    bgl.glEnd()  
      
    if LINE_TYPE == "GL_LINE_STIPPLE":  
        bgl.glDisable(bgl.GL_LINE_STIPPLE)  
        bgl.glEnable(bgl.GL_BLEND)  # back to uninterupted lines  
        bgl.glLineWidth(1)
    return
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

def draw_3d_text(context, pt, msg, size):
    '''
    draw text at a given pt
    '''
    point_2d = location_3d_to_region_2d(context.region, context.space_data.region_3d, pt)

    bgl.glColor4f(0.9, 0.9, 0.9, 0.8)
    font_id = 0  # XXX, need to find out how best to get this.

    # draw some text
    blf.position(font_id, point_2d[0], point_2d[1], 0)
    blf.size(font_id, size, 72)
    blf.draw(font_id, msg)
    
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


class WatershedObjectCurvature(bpy.types.Operator):
    """Roll Water Drops to find Cusps"""
    bl_idname = "view3d.watershed_cusp_finder"
    bl_label = "Find Cusps"

    
    def draw_callback_water_drops(self, context):
        
        mx = context.object.matrix_world
        
        draw_3d_points(context, [self.com], (1,.1,1,.5), 6)
        if not self.consensus_generated:
            for droplet in self.drops:
                vs = [mx * self.bme.verts[i].co for i in droplet.ind_path]
            
                #draw_3d_points(context, vs, (.2,.3,.8,1), 2)
                draw_3d_points(context, [vs[-1]], (1,.3,.3,1), 3)
                #draw_3d_points(context, [vs[0]], (.3,1,.3,1), 4)
        
        if self.consensus_generated:
            
            vs = [mx * self.bme.verts[i].co for i in self.consensus_list]
            draw_3d_points(context, vs, (.2,.8,.8,1), 5)
            
        if self.sorted_by_value:
            vs = [mx * self.bme.verts[i].co for i in self.best_verts]
            draw_3d_points(context, vs, (.8,.8,.2,1), 3)
        
        
        if len(self.clipped_verts):
            vs = [mx * v for v in self.clipped_verts]
            draw_3d_points(context, vs, (1,.3,1,1), 4)
            
            
        if len(self.bez_curve):
            vs = [mx * v for v in self.bez_curve]
            draw_polyline_from_3dpoints(context, vs, (.2,1,.2,1), 3)
            draw_3d_points(context, vs, (.2,1,.2,1), 5)
        #if len(self.polyline):
            #draw_polyline_from_3dpoints(context, self.polyline, (1,1,.2,1), 2)
            
        #    for i, v in enumerate(self.polyline):
        #        msg = str(i)
        #        draw_3d_text(context, v, msg, 20)
                         
    def roll_droplets(self,context):
        count_up = 0
        count_dn = 0
        for drop in self.drops:
            #if not drop.peaked:
            #    drop.roll_uphill()
            #    count_up += 1
            if not drop.settled:
                drop.roll_downhill()
                count_dn += 1
                
        return count_dn

    def build_concensus(self,context):
        
        list_inds = [drop.dn_vert.index for drop in self.drops]
        vals = [drop.dnH for drop in self.drops]
        
        
        unique = set(list_inds)
        unique_vals = [vals[list_inds.index(ind)] for ind in unique]
        
        
        print('there are %i droplets' %len(list_inds))
        print('ther are %i unique maxima' % len(unique))
    
        best = Counter(list_inds)
        
        consensus_tupples = best.most_common(self.consensus_count)
        self.consensus_list = [tup[0] for tup in consensus_tupples]
        self.consensus_dict = {}  #throw it all away?
        
        #map consensus to verts.  Later we will merge into this dict
        for tup in consensus_tupples:
            self.consensus_dict[tup[0]] = tup[1]
            
        #print(self.consensus_list)
        self.consensus_generated = True
        
    
    def sort_by_value(self,context):
        
        list_inds = [drop.dn_vert.index for drop in self.drops]
        vals = [drop.dnH for drop in self.drops]
        
        
        unique_inds = list(set(list_inds))
        unique_vals = [vals[list_inds.index(ind)] for ind in unique_inds]
        
        bme_inds_by_val = [i for (v,i) in sorted(zip(unique_vals, unique_inds))]
        self.best_verts = bme_inds_by_val[0:self.consensus_count]
        self.sorted_by_value = True
    
    
    def merge_close_consensus_points(self):
        '''
        cusps usually aren't closer than 2mm
        actually we aren't merging, we just toss the one with less votes
        '''
        
        #consensus list is sorted with most voted for locations first
        #start at back of list and work forward
        to_remove = []
        new_verts = []
        l_co = [self.bme.verts[i].co for i in self.consensus_list]
        N = len(l_co)
        for i, pt in enumerate(l_co):
            
            #if i in to_remove:
            #    continue
            
            ds, inds, vs = points_within_radius(pt, l_co, 7)
            
            if len(vs):
                new_co = Vector((0,0,0))
                for v in vs:
                    new_co += v
                new_co += pt
                new_co *= 1/(len(vs) + 1)
            else:
                new_co = pt
                
            new_verts.append(new_co)
                        
            for j in inds:
                if j > i:
                    to_remove.append(j)  
               
            
        to_remove = list(set(to_remove))
        to_remove.sort(reverse = True)
        
        print('removed %i too close consensus points' % len(to_remove))
        print(to_remove)
        for n in to_remove:
            l_co.pop(n)
            
        
        
        self.clipped_verts = new_verts
        
        return
        
    def fit_cubic_consensus_points(self):
        '''
        let i's be indices in the actual bmesh
        let j's be arbitrary list comprehension indices
        let n's be the incidices in our consensus point lists range 0,len(consensus_list)
        '''
        l_co = self.clipped_verts
        
        
        com, no = calculate_plane(l_co)  #an easy way to estimate occlusal plane
        no.normalize()
        
        
        #neigbors = set(l_co)
        box = bbox(l_co)
        
        diag = (box[1]-box[0])**2 + (box[3]-box[2])**2 + (box[5]-box[4])**2
        diag = math.pow(diag,.5)
        
        #neighbor_path = [neighbors.pop()]
        
        #establish a direction
        #n, v, d  = closest_point(neighbor_path[0], list(neighbors))
        #if d < .2 * diag:
        #    neighbor_path.append(v)
        
        #ended = Fase
        #while len(neighbors) and not ended:   
        #   n, v, d  = closest_point(neighbor_path[0], list(neighbors)
        
        #flattened spokes
        rs = [v - v.dot(no)*v - com for v in l_co]
        
        R0 = rs[random.randint(0,len(rs)-1)]
        
        theta_dict = {}
        thetas = []
        for r, v in zip(rs,l_co):
            angle = r.angle(R0)
            
            if r != R0:
                rno = r.cross(R0)
                if rno.dot(no) < 0:
                    angle *= -1
                    angle += 2 * math.pi
            
            theta_dict[round(angle,4)] = v
            thetas.append(round(angle,4))
        
        print(thetas)
        thetas.sort()
        print(thetas)
        diffs = [thetas[i]-thetas[i-1] for i in range(0,len(thetas))]
        n = diffs.index(max(diffs)) # -1
        theta_shift = thetas[n:] + thetas[:n]
        
        self.polyline = [theta_dict[theta] for theta in theta_shift]
        #inds_in_order = [theta_dict[theta] for theta in thetas]
        #self.polyline = [l_co[i] for i in inds_in_order]
        
        self.com = com

        l_bpts = cubic_bezier_fit_points(self.polyline, 1, depth=0, t0=0, t3=1, allow_split=True, force_split=False)
        self.bez_curve = []
        N = 20
        for i,bpts in enumerate(l_bpts):
            t0,t3,p0,p1,p2, p3 = bpts
            

            new_pts = [cubic_bezier_blend_t(p0,p1,p2,p3,i/N) for i in range(0,N)]
            
            self.bez_curve.extend(new_pts)  
             
    def invoke(self,context, event):
        ob = bpy.context.object
        self.bme = bmesh.new()
        self.bme.from_mesh(ob.data)

        self.bme.verts.ensure_lookup_table()
        self.bme.faces.ensure_lookup_table()
        
        curv_id = self.bme.verts.layers.float['max_curve']
        
        rand_sample = list(set([random.randint(0,len(self.bme.verts)-1) for i in range(math.floor(.05 * len(self.bme.verts)))]))
        
        sel_verts = [self.bme.verts[i] for i in rand_sample]
        pln_pt = Vector((0,0,0))  #TODO calc centroid of high curvature
        pln_no = Vector((0,0,1))  #TODO....calc this from PCA analysis of clustered curvature
        self.drops = [CuspWaterDroplet(v, pln_pt, pln_no, curv_id) for v in sel_verts]
        
        self.consensus_count = 10
        self.consensus_list = []
        self.consensus_dict = {}
        self.consensus_generated = False
        self.bez_curve = []
        self.polyline = []
        self.clipped_verts = []
        self.com = Vector((0,0,0))
        
        
        self.best_verts = []
        self.sorted_by_value = False
        
        self._handle = bpy.types.SpaceView3D.draw_handler_add(self.draw_callback_water_drops, (context,), 'WINDOW', 'POST_PIXEL')
        context.window_manager.modal_handler_add(self)
    
        return {'RUNNING_MODAL'}
    
    def modal(self,context,event):
        context.area.tag_redraw()
        
        if event.type == 'RET' and event.value == 'PRESS':
            for drop in self.drops:
                for i in drop.ind_path:
                    self.bme.verts[i].select = True
            
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self.bme.to_mesh(context.object.data)
            self.bme.free()
            
            return {'FINISHED'}
          
        
        
        elif event.type == 'Q' and event.value == 'PRESS':
            n_rolling = self.roll_droplets(context)
            
            iters = 0
            while n_rolling > 5 and iters < 400:
                n_rolling = self.roll_droplets(context)
                iters += 1
                
            if iters >= 399:
                print('too much rolling')    
            
            self.consensus_count = 20    
            self.build_concensus(context)
            
            l_co = [self.bme.verts[i].co for i in self.consensus_list]
            test_no = vector_average([self.bme.verts[i].normal for i in self.consensus_list])
            test_no.normalize()
            pt, pno = calculate_plane(l_co)
            
            
            if pno.dot(test_no) < 0:
                pno *= -1
            
            self.pln_pt = pt - 5*pno
            self.pln_no = pno
                
            mx = context.object.matrix_world
            imx = mx.inverted()
            no_mx = mx.transposed().inverted().to_3x3()
            
            
            Z = no_mx * pno
            loc = mx * pt - 5 * Z
            
            ob_y = no_mx * Vector((0,1,0))
            X = ob_y.cross(Z)
            Y = Z.cross(X)
            
            Z.normalize()
            Y.normalize()
            X.normalize()
            
            wmx = Matrix.Identity(4)
            wmx[0][0], wmx[1][0], wmx[2][0] = X[0], X[1], X[2]
            wmx[0][1], wmx[1][1], wmx[2][1] = Y[0], Y[1], Y[2]
            wmx[0][2], wmx[1][2], wmx[2][2] = Z[0], Z[1], Z[2]
            wmx[0][3], wmx[1][3], wmx[2][3] = loc[0], loc[1], loc[2]
            
            #circ_bm = bmesh.new()
            #bmesh.ops.create_circle(circ_bm, cap_ends = True, cap_tris = False, segments = 10, diameter = .5 *min(context.object.dimensions) + .5 *max(context.object.dimensions))
            
            # Finish up, write the bmesh into a new mesh
            #me = bpy.data.meshes.new("Occlusal Plane")
            #circ_bm.to_mesh(me)
            #circ_bm.free()

            # Add the mesh to the scene
            #scene = bpy.context.scene
            #obj = bpy.data.objects.new("Object", me)
            #scene.objects.link(obj)
            #obj.matrix_world = wmx
            return {'RUNNING_MODAL'}
        
        
        elif event.type == 'W' and event.value == 'PRESS':
            curv_id = self.bme.verts.layers.float['max_curve']
            
            start = time.time()
            cut_geom = self.bme.faces[:] + self.bme.verts[:] + self.bme.edges[:]
            bmesh.ops.bisect_plane(self.bme, geom = cut_geom, dist = .000001, plane_co = self.pln_pt, plane_no = self.pln_no, use_snap_center = False, clear_outer=False, clear_inner=True)
            self.bme.verts.ensure_lookup_table()
            self.bme.faces.ensure_lookup_table()
            
            
            rand_sample = list(set([random.randint(0,len(self.bme.verts)-1) for i in range(math.floor(.2 * len(self.bme.verts)))]))
            self.drops = [CuspWaterDroplet(self.bme.verts[i], self.pln_pt, self.pln_no, curv_id) for i in rand_sample]
            dur = time.time() - start
            print('took %f seconds to cut the mesh and generate drops' % dur)
            
            start = time.time()
            n_rolling = self.roll_droplets(context)
            iters = 0
            while n_rolling > 10 and iters < 100:
                n_rolling = self.roll_droplets(context)
                iters += 1
            
            self.consensus_count = 80
            self.build_concensus(context)
            
            dur = time.time() - start
            print('took %f seconds to roll the drops' % dur)
            return {'RUNNING_MODAL'}
               
        elif event.type == 'UP_ARROW' and event.value == 'PRESS':
            n_rolling = self.roll_droplets(context)
            
            iters = 0
            while n_rolling > 10 and iters < 100:
                n_rolling = self.roll_droplets(context)
                iters += 1
            return {'RUNNING_MODAL'}
        
        
        elif event.type == 'LEFT_ARROW' and event.value == 'PRESS':
            self.consensus_count -= 5
            self.build_concensus(context)
            return {'RUNNING_MODAL'}
        
        elif event.type == 'RIGHT_ARROW' and event.value == 'PRESS':
            self.consensus_count += 5
            self.build_concensus(context)
            return {'RUNNING_MODAL'}
        
        elif event.type == 'C' and event.value == 'PRESS':
            self.build_concensus(context) 
            return {'RUNNING_MODAL'}
        
        elif event.type == 'M' and event.value == 'PRESS':
            self.merge_close_consensus_points()
            return {'RUNNING_MODAL'}
            
        elif event.type == 'B' and event.value == 'PRESS' and self.consensus_generated:
            self.fit_cubic_consensus_points()
            return {'RUNNING_MODAL'}
            
        elif event.type == 'S' and event.value == 'PRESS':
            self.sort_by_value(context)
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
    bpy.utils.register_class(WatershedObjectCurvature)
    
def unregister():
    bpy.utils.unregister_class(ViewOperatorObjectCurve)
    bpy.utils.unregister_class(ViewObjectSalience)
    bpy.utils.unregister_class(WatershedObjectCurvature)
    

if __name__ == "__main__":
    register()

