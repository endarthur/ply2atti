#!/usr/bin/python
# -*- coding: utf-8 -*-

# MIT License

# Copyright (c) 2017 Arthur Endlein

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function

import numpy as np
from networkx import Graph, connected_components
import xlsxwriter


def calc_sphere(x, y, z):
    """Calculate spherical coordinates for axial data."""
    return np.degrees(np.arctan2(*(np.array((
        x, y)) * np.sign(z)))) % 360, np.degrees(np.arccos(np.abs(z)))


def general_axis(data, order=0):
    """Calculates the Nth eigenvector dataset tensor, first one by default."""
    direction_tensor = np.cov(data.T[:3, :])
    # print direction_tensor
    eigen_values, eigen_vectors = np.linalg.eigh(direction_tensor, UPLO='U')
    eigen_values_order = eigen_values.argsort()[::-1]
    cone_axis = eigen_vectors[:, eigen_values_order[order]]
    return cone_axis / np.linalg.norm(cone_axis)


def calibrate_azimuth(data, target_color, target_azimuth):
    calibrate_data = np.mean(data[target_color], axis=0)
    d_az = target_azimuth - calibrate_data[0]
    for color in data.keys():
        data[color] = [((az + d_az) % 360, dip) for az, dip in data[color]]
    return data


def load_ply(f):
    properties = vertex_properties = []
    face_properties = []
    line = b""
    while b"end_header" not in line:
        line = f.readline()
        if line.startswith(b"element vertex"):
            vertex_n = int(line.split()[-1])
        if line.startswith(b"element face"):
            face_n = int(line.split()[-1])
            properties = face_properties
        if line.startswith(b"property"):
            properties.append(line.split()[-1].strip())
    vertex_dtype = [
        ("position", np.float32, 3),
    ]
    vertex_dtype += [
        ("normal", np.float32, 3),
    ] if b"nx" in vertex_properties else []
    vertex_dtype += [
        ("color", np.uint8, 4),
    ] if b"alpha" in vertex_properties else [
        ("color", np.uint8, 3),
    ]
    faces_dtype = [
        ("face_n", np.uint8, 1),
        ("indices", np.int32, 3),
    ]
    faces_dtype += [
        ("color", np.uint8, 4),
    ] if b"alpha" in face_properties else [
        ("color", np.uint8, 3),
    ] if b"red" in face_properties else []
    vertices = np.fromfile(f, dtype=vertex_dtype, count=vertex_n)
    faces = np.fromfile(f, dtype=faces_dtype, count=face_n)
    return vertices, faces


def extract_colored_faces(fname, colors):
    output = {color: [] for color in colors}
    vertices, faces = load_ply(fname)

    for color in colors:
        colored_vertices_indices = np.nonzero((vertices['color'] == color).all(
            axis=1))[0]

        v0 = np.in1d(faces["indices"][:, 0], colored_vertices_indices)
        v1 = np.in1d(faces["indices"][:, 1], colored_vertices_indices)
        v2 = np.in1d(faces["indices"][:, 2], colored_vertices_indices)

        # colored_faces = np.nonzero(
        #     np.all(
        #         (np.in1d(faces["indices"][:, 0], colored_vertices_indices),
        #          np.in1d(faces["indices"][:, 1], colored_vertices_indices),
        #          np.in1d(faces["indices"][:, 2], colored_vertices_indices)),
        #         axis=0))[0]

        # colored_faces_graph = Graph()
        # colored_faces_graph.add_edges_from(
        #     faces['indices'][colored_faces][:, :2])
        # colored_faces_graph.add_edges_from(
        #     faces['indices'][colored_faces][:, 1:])
        # colored_faces_graph.add_edges_from(
        #     faces['indices'][colored_faces][:, (0, 2)])

        colored_faces_graph = Graph()
        colored_faces_graph.add_edges_from(
            faces[np.logical_and(v0, v1)]['indices'][:, :2])
        colored_faces_graph.add_edges_from(
            faces[np.logical_and(v1, v2)]['indices'][:, 1:])
        colored_faces_graph.add_edges_from(
            faces[np.logical_and(v0, v2)]['indices'][:, (0, 2)])

        planes_vertices_indices = list(
            connected_components(colored_faces_graph))
        for plane_vertices_indices in planes_vertices_indices:
            colored_vertices = vertices["position"][list(
                plane_vertices_indices)]
            dipdir, dip = calc_sphere(*general_axis(colored_vertices, -1))
            X, Y, Z = colored_vertices.mean(axis=0)
            highest_vertex = colored_vertices[np.argmax(
                colored_vertices[:, 2]), :]
            lowest_vertex = colored_vertices[np.argmin(colored_vertices[:,
                                                                        2]), :]
            trace = np.linalg.norm(highest_vertex - lowest_vertex)
            output[color].append((dipdir, dip, X, Y, Z, trace))
    return output


def extract_colored_point_clouds(fname, colors):
    output = {color: [] for color in colors}
    vertices, faces = load_ply(fname)
    for color in colors:
        colored_vertices_indices = (vertices['color'] == color).all(
            axis=1)
        colored_vertices = vertices["position"][colored_vertices_indices]
        dipdir, dip = calc_sphere(*general_axis(colored_vertices, -1))
        X, Y, Z = colored_vertices.mean(axis=0)
        highest_vertex = colored_vertices[np.argmax(colored_vertices[:, 2]), :]
        lowest_vertex = colored_vertices[np.argmin(colored_vertices[:, 2]), :]
        trace = np.linalg.norm(highest_vertex - lowest_vertex)
        output[color].append((dipdir, dip, X, Y, Z, trace))
    return output


class Ply2AttiWorkbook:
    def __init__(self, filename, color=False, atti_format='0.0'):
        self.wb = xlsxwriter.Workbook(filename)
        self.wa = self.wb.add_worksheet("attitudes")
        self.wc = self.wb.add_worksheet("coordinates")
        self.wa.write(0, 0, "dip direction")
        self.wa.write(0, 1,  "dip")
        self.wc.write(0, 0, "X")
        self.wc.write(0, 1, "Y")
        self.wc.write(0, 2, "Z")
        self.wc.write(0, 3, "attitude")
        self.wc.write(0, 4, "trace")

        if color:
            self.wa.write(0, 2, "color")
            self.wc.write(0, 5, "color")

        self.i = 1

        self.atti_format = self.wb.add_format({'num_format': atti_format})

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.wb.close()

    def write_data(self, dd, d, x, y, z, trace, color=None):
        self.wa.write(self.i, 0, dd, self.atti_format)
        self.wa.write(self.i, 1, d, self.atti_format)

        self.wc.write(self.i, 0, x)
        self.wc.write(self.i, 1, y)
        self.wc.write(self.i, 2, z)
        self.wc.write(self.i, 3, f"{dd:03.0f}/{d:02.0f}")
        self.wc.write(self.i, 4, trace)

        if color is not None:
            self.wa.write(self.i, 2, f"{color}")
            self.wc.write(self.i, 5, f"{color}")

        self.i += 1


def main():
    import argparse
    from os import path
    from datetime import datetime
    parser = argparse.ArgumentParser(
        description=
        "Extract planes painted on stanford polygon (.ply) files on meshlab.")
    parser.add_argument(
        "--file",
        action="store",
        dest="infile",
        required=True,
        help="input painted 3d model")
    parser.add_argument(
        "--split",
        action="store_true",
        default=False,
        dest="split",
        help="Split the resulting attitudes in one file per color")
    parser.add_argument(
        "--xlsx",
        action="store_true",
        default=False,
        dest="xlsx",
        help="Export the resulting data to a .xlsx file instead of csv")
    parser.add_argument(
        "--format",
        action="store",
        default='0.0',
        dest="atti_format",
        required=False,
        help="""Format definition for attitudes in .xlsx file. Check \
        http://bit.ly/2K9R1aG for more details, defaults to '0.0'.""")
    # parser.add_argument(
    #     "--pointcloud",
    #     action="store_true",
    #     default=False,
    #     dest="pointcloud",
    #     help=
    #     "Export all normals to vertices of each plane instead of calculated
    # average attitudes"
    # )
    parser.add_argument("colors", nargs="+", help="Colors to be extracted")
    args = parser.parse_args()
    colors = []
    starttime = datetime.now()
    for color in args.colors:
        components = tuple(color.split(','))
        if len(components) < 4:
            components += ('255', )
        colors.append(tuple([int(component) for component in components]))
    filename = path.splitext(args.infile)[0]
    with open(args.infile, 'rb') as f:
        output = extract_colored_faces(f, colors)
    if args.split:
        if not args.xlsx:
            for color in output.keys():
                with open("{0}_{1}.txt".format(
                        filename, color), 'w') as f, open(
                        "{0}_{1}_coords.txt".format(filename, color),
                        'w') as coordf:
                    coordf.write("X\tY\tZ\tatti\ttrace\n")
                    for dipdir, dip, X, Y, Z, trace in output[color]:
                        f.write("{0}\t{1}\n".format(dipdir, dip))
                        coordf.write("{0}\t{1}\t{2}\t{3}/{4}\t{5}\n".format(
                            X, Y, Z, int(dipdir), int(dip), trace))
        else:
            for color in output.keys():
                with Ply2AttiWorkbook(
                        f"{filename}_{color}.xlsx",
                        atti_format=args.atti_format) as f:
                    for dd, d, x, y, z, trace in output[color]:
                        f.write_data(dd, d, x, y, z, trace)
    else:
        if not args.xlsx:
            with open("{0}_attitudes.txt".format(filename), 'w') as f, open(
                    "{0}_coords.txt".format(filename), 'w') as coordf:
                coordf.write("X\tY\tZ\tatti\ttrace\n")
                for color in output.keys():
                    f.write("#{0}\n".format(color))
                    coordf.write("#{0}\n".format(color))
                    for dipdir, dip, X, Y, Z, trace in output[color]:
                        f.write("{0}\t{1}\n".format(dipdir, dip))
                        coordf.write("{0}\t{1}\t{2}\t{3}/{4}\t{5}\n".format(
                            X, Y, Z, int(dipdir), int(dip), trace))
        else:
            with Ply2AttiWorkbook(
                    f"{filename}.xlsx", color=True,
                    atti_format=args.atti_format) as f:
                for color in output.keys():
                    for dd, d, x, y, z, trace in output[color]:
                        f.write_data(dd, d, x, y, z, trace, color)
    print("Total time processing {}.".format(datetime.now() - starttime))
    # print("\a")


if __name__ == "__main__":
    main()
