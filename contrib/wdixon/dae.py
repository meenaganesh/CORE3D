from collada import source, material, geometry, scene, Collada
import numpy
from plyfile import PlyData


ply = PlyData.read(r'/raid/data/wdixon/render/crop1_meshlab_vn.ply')
vertices_xyz, vertices_normals = numpy.hsplit(numpy.array(ply.elements[0].data.tolist()), 2)
vertices_xyz[:,0] = vertices_xyz[:,0] - vertices_xyz[0,0]
vertices_xyz[:,1] = vertices_xyz[:,1] - vertices_xyz[0,1]
vertices_xyz[:,2] = vertices_xyz[:,2] - vertices_xyz[:,2].min()

vertices = source.FloatSource("mesh_verts-array", vertices_xyz.flatten(), ('X', 'Y', 'Z'))


normals = source.FloatSource("mesh_normals-array", vertices_normals.flatten(), ('X', 'Y', 'Z'))

x = ply.elements[1].data.tolist()
faces = numpy.ndarray(shape=(len(x)*3,2), dtype=int)

faces[:,0] = numpy.array(x).flatten()
faces[:,1] = numpy.array(x).flatten()


mesh = Collada()

effect = material.Effect("effect0", [], "phong", double_sided=True, diffuse=(1, 0, 0), specular=(0, 1, 0))
mat = material.Material("material0", "mymaterial", effect)
mesh.effects.append(effect)
mesh.materials.append(mat)

geom = geometry.Geometry(mesh, "geometry0", "core3d0", [vertices, normals])
input_list = source.InputList()
input_list.addInput(0, 'VERTEX', "#mesh_verts-array")
input_list.addInput(1, 'NORMAL', "#mesh_normals-array")


triset = geom.createTriangleSet(faces.flatten(), input_list, "materialref")
geom.primitives.append(triset)
mesh.geometries.append(geom)

rotate = scene.RotateTransform(1, 0, 0, -90)
matnode = scene.MaterialNode("materialref", mat, inputs=[])
geomnode = scene.GeometryNode(geom, [matnode])
node = scene.Node("node0", children=[geomnode], transforms=[rotate])


myscene = scene.Scene("myscene", [node])
mesh.scenes.append(myscene)
mesh.scene = myscene



mesh.write('/home/wdixon/test_mesh_rot4.dae')
