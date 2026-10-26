"""A real GPU check: imported/static room front faces must face the camera.

Stock K2 001ebo1 also has positive cross(face edges) dot supplied normals.
No game files or live UI are needed for this small rasterization contract.
"""
import numpy as np
import pytest


@pytest.mark.parametrize("elevation,visible", [(60, True), (-60, False)])
def test_static_floor_renders_front_and_culls_back(elevation, visible):
    from src.adapters.rendering.moderngl_legacy_bridge import GpuRenderer
    from src.core.camera.arcball_camera import ArcBallCamera
    from src.core.geometry.model_data import KotorModel, ModelNode, NodeFlags

    renderer = GpuRenderer()
    if not renderer._ensure_context():
        pytest.skip("A real ModernGL context is required")
    try:
        renderer.viewport_background = (0., 0., 0.)
        renderer.cull_faces = True
        floor = ModelNode(name="floor", flags=int(NodeFlags.MESH) | 1,
                          vertices=[(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)],
                          normals=[(0,0,1)]*4, faces=[(0,1,2),(0,2,3)],
                          uvs=[(0,0),(1,0),(1,1),(0,1)],
                          diffuse=(1,1,1), selfillum=(1,1,1))
        model = KotorModel(name="culling_fixture", root_node=floor)
        camera = ArcBallCamera()
        camera.target=[0,0,0];camera.distance=4
        camera.azimuth=45;camera.elevation=elevation
        frame=renderer.render(model,camera,128,128)
        assert frame is not None
        # Ignore the backend's dim floor grid; count the bright fixture surface.
        foreground=int(np.count_nonzero(np.asarray(frame)[...,:3].min(axis=2)>160))
        assert (foreground > 1000) if visible else (foreground == 0)
    finally:
        renderer.release()
