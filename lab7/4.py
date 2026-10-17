import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print(" ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def colmap_available():
    return shutil.which("colmap") is not None


def run_cli(images, workspace, use_gpu, quality, dense):
    db = workspace / "database.db"
    sparse = workspace / "sparse"
    dense_dir = workspace / "dense"
    sparse.mkdir(parents=True, exist_ok=True)
    gpu = "1" if use_gpu else "0"

    run(["colmap", "feature_extractor",
         "--database_path", db,
         "--image_path", images,
         "--ImageReader.single_camera", "1",
         "--ImageReader.camera_model", "SIMPLE_RADIAL",
         "--SiftExtraction.use_gpu", gpu])

    run(["colmap", "exhaustive_matcher",
         "--database_path", db,
         "--SiftMatching.use_gpu", gpu])

    run(["colmap", "mapper",
         "--database_path", db,
         "--image_path", images,
         "--output_path", sparse])

    model = sparse / "0"
    if not model.exists():
        raise SystemExit("Reconstruction failed: no model produced. Use more overlapping images.")

    run(["colmap", "model_converter",
         "--input_path", model,
         "--output_path", workspace / "sparse_points.ply",
         "--output_type", "PLY"])

    run(["colmap", "model_converter",
         "--input_path", model,
         "--output_path", workspace / "sparse_txt",
         "--output_type", "TXT"])

    if not dense:
        return workspace / "sparse_points.ply", None

    dense_dir.mkdir(parents=True, exist_ok=True)
    run(["colmap", "image_undistorter",
         "--image_path", images,
         "--input_path", model,
         "--output_path", dense_dir,
         "--output_type", "COLMAP"])

    run(["colmap", "patch_match_stereo",
         "--workspace_path", dense_dir,
         "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "true",
         "--PatchMatchStereo.max_image_size", "2000" if quality == "high" else "1000"])

    fused = dense_dir / "fused.ply"
    run(["colmap", "stereo_fusion",
         "--workspace_path", dense_dir,
         "--workspace_format", "COLMAP",
         "--input_type", "geometric",
         "--output_path", fused])

    mesh = dense_dir / "meshed-poisson.ply"
    run(["colmap", "poisson_mesher",
         "--input_path", fused,
         "--output_path", mesh])

    return fused, mesh


def run_pycolmap(images, workspace):
    import pycolmap

    db = workspace / "database.db"
    sparse = workspace / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)

    pycolmap.extract_features(db, images)
    pycolmap.match_exhaustive(db)
    maps = pycolmap.incremental_mapping(db, images, sparse)
    if not maps:
        raise SystemExit("Reconstruction failed: no model produced")
    best = max(maps.values(), key=lambda r: r.num_reg_images())
    print(best.summary())
    ply = workspace / "sparse_points.ply"
    best.export_PLY(ply)
    return ply, None


def build_mesh_open3d(point_cloud_path, mesh_path, depth=9):
    import open3d as o3d
    import numpy as np

    pcd = o3d.io.read_point_cloud(str(point_cloud_path))
    if len(pcd.points) == 0:
        raise SystemExit("Point cloud is empty")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.05))
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    print("Mesh written to", mesh_path)
    return pcd, mesh


def visualize(pcd_path, mesh_path):
    import open3d as o3d

    geometries = []
    if pcd_path and Path(pcd_path).exists():
        geometries.append(o3d.io.read_point_cloud(str(pcd_path)))
    if mesh_path and Path(mesh_path).exists():
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_vertex_normals()
        geometries.append(mesh)
    if geometries:
        o3d.visualization.draw_geometries(geometries, window_name="COLMAP 3D scene")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--workspace", default="colmap_workspace")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--quality", choices=["medium", "high"], default="medium")
    parser.add_argument("--backend", choices=["auto", "cli", "pycolmap"], default="auto")
    parser.add_argument("--no-view", action="store_true")
    args = parser.parse_args()

    images = Path(args.images).resolve()
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    if not images.is_dir() or not any(images.iterdir()):
        raise SystemExit("Image folder is missing or empty")

    backend = args.backend
    if backend == "auto":
        backend = "cli" if colmap_available() else "pycolmap"

    if backend == "cli":
        if not colmap_available():
            raise SystemExit("colmap executable not found in PATH")
        cloud, mesh = run_cli(images, workspace, args.gpu, args.quality, args.dense)
    else:
        try:
            cloud, mesh = run_pycolmap(images, workspace)
        except ImportError:
            raise SystemExit("Install COLMAP or run: pip install pycolmap")

    if mesh is None:
        try:
            mesh = workspace / "mesh.ply"
            build_mesh_open3d(cloud, mesh)
        except ImportError:
            print("open3d not installed; skipping mesh. Run: pip install open3d")
            mesh = None

    print("Point cloud:", cloud)
    print("Mesh:", mesh)

    if not args.no_view:
        try:
            visualize(cloud, mesh)
        except ImportError:
            pass


if __name__ == "__main__":
    sys.exit(main())