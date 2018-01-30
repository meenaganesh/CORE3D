import logging
import os
import numpy
from simplekml import Kml, Model, AltitudeMode, Orientation, Scale, Location
from collada import source, material, geometry, scene, Collada
from plyfile import PlyData
from osgeo.osr import SpatialReference, CoordinateTransformation


class CrsUtil(object):
    def __init__(self, src_crs, dst_crs):
        self.src_crs = SpatialReference()
        self.src_crs.ImportFromEPSG(src_crs)
        self.dst_crs = SpatialReference()
        self.dst_crs.ImportFromEPSG(dst_crs)
        self.src_to_dst = CoordinateTransformation(self.src_crs, self.dst_crs)

    def transform(self, src_x, src_y):
        dest_x, dest_y, dest_z = self.src_to_dst.TransformPoint(src_x, src_y)
        return dest_x, dest_y


class ColladaModel(object):

    def __init__(self, name, kml_file):
        # Create the KML document
        self.kml = Kml(name=name, open=1)
        self.kml_file = kml_file
        self.model_dir = os.path.join(os.path.dirname(kml_file), "models")
        os.makedirs(self.model_dir, exist_ok=True)
#        self.crs = CrsUtil(3857, 4326)
        self.crs = CrsUtil(32617, 4326)

    def add_model(self, id, in_file: str):
        ply = PlyData.read(in_file)
        vertices_xyz, vertices_normals = numpy.hsplit(numpy.array(ply.elements[0].data.tolist()), 2)
        x = vertices_xyz[0, 0]
        y = vertices_xyz[0, 1]
        lng, lat = self.crs.transform(x,y)

        vertices_xyz[:, 0] = vertices_xyz[:, 0] - vertices_xyz[0, 0]
        vertices_xyz[:, 1] = vertices_xyz[:, 1] - vertices_xyz[0, 1]
        vertices_xyz[:, 2] = vertices_xyz[:, 2] - vertices_xyz[:, 2].min()

        vertices = source.FloatSource("mesh_verts-array", vertices_xyz.flatten(), ('X', 'Y', 'Z'))
        normals = source.FloatSource("mesh_normals-array", vertices_normals.flatten(), ('X', 'Y', 'Z'))

        x = ply.elements[1].data.tolist()
        faces = numpy.ndarray(shape=(len(x) * 3, 2), dtype=int)

        faces[:, 0] = numpy.array(x).flatten()
        faces[:, 1] = numpy.array(x).flatten()

        mesh = Collada()

        # supported = ['emission', 'ambient', 'diffuse', 'specular',
        #              'shininess', 'reflective', 'reflectivity',
        #              'transparent', 'transparency', 'index_of_refraction']
        # """Supported material properties list."""
        # shaders = ['phong', 'lambert', 'blinn', 'constant']

        effect = material.Effect("effect0", [], "lambert", double_sided=True, diffuse=(1, 1, 1),
                                 specular=(0, 1, 0))
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

        rotatez = scene.RotateTransform(0, 0, 1, -180)
        rotate = scene.RotateTransform(1, 0, 0, -90)
        matnode = scene.MaterialNode("materialref", mat, inputs=[])
        geomnode = scene.GeometryNode(geom, [matnode])
        node = scene.Node("node0", children=[geomnode], transforms=[rotate, rotatez])

        myscene = scene.Scene("myscene", [node])
        mesh.scenes.append(myscene)
        mesh.scene = myscene

        self.add_dae(mesh, id, lng, lat)

    def add_dae(self, mesh, id, lng, lat):
        dae = id + ".dae"
        model_path = os.path.join(self.model_dir, dae)
        mesh.write(model_path)

        scale = 1.0
        model = self.kml.newmodel(name="Building_"+str(id), altitudemode=AltitudeMode.clamptoground,
                                  description="building description")
        model.location = Location(longitude=lng, latitude=lat)
        path = self.kml.addfile(model_path)
        model.link.href = path

        # # Turn-off default icon and text and hide the linestring
        # model.iconstyle.icon.href = ""
        # model.labelstyle.scale = 0
        # model.linestyle.width = 0

    def close(self):
        # Saving
        self.kml.savekmz(self.kml_file)


if __name__ == "__main__":
    # Below is an example of how to use this AOI pipeline

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("aoi.log"),
            logging.StreamHandler()
        ])

    m = ColladaModel("build", '/home/wdixon/test_building/buildings1.kmz')
    # m.add_model("my_bldg",
    #             '/home/wdixon/test_simplify.ply')  # jacksonville_dhm_cut_ascii.ply') # '/home/wdixon/jacksonville_dhm_cut_trim.ply')  #/home/wdixon/crop1_meshlab_vn.ply'

    m.add_model('my_bldg_1', '/home/wdixon/building_1.ply')
    m.add_model('my_bldg_2', '/home/wdixon/building_2.ply')
    m.add_model('my_bldg_5', '/home/wdixon/building_5.ply')

    m.close()
