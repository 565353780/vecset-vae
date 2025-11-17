import numpy as np
import open3d as o3d

from vecset_vae.Module.detector import Detector


def demo():
    model_file_path = "./output/vecset-64x64-v1/model_last.pth"
    use_ema = False
    batch_size = 120000
    resolution = 64
    device = "cuda"

    detector = Detector(
        model_file_path,
        use_ema,
        batch_size,
        resolution,
        device,
    )

    mesh = detector.detectDataset(0)

    data_dict = detector.tsdf_dataset.__getitem__(0)

    coarse_surface = data_dict["coarse_surface"].numpy().astype(np.float64)[:, :3]

    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(coarse_surface)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.faces, dtype=np.int32)

    recon_mesh = o3d.geometry.TriangleMesh()
    recon_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    recon_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    recon_mesh.compute_vertex_normals()

    print(recon_mesh)

    gt_pcd.translate([-2, 0, 0])
    o3d.visualization.draw_geometries([recon_mesh, gt_pcd])
    return True
