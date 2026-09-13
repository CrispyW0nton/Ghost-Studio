"""Legacy generic mesh converters.

OBJ <-> KotorModel, FBX -> KotorModel, KotorModel -> OBJ/FBX, and TGA <-> TPC
texture conversion.

Legacy Generic Mesh FBX Importer - NOT FOR ANIMATION PIPELINE

Fallback chain: pyassimp -> assimp_py -> trimesh.
Purpose: geometry conversion for legacy mesh workflows.
Output: static mesh/model data; this path is not the Retarget Workbench
``SourceSkeletonClip`` animation pipeline.

IMPORTANT: not used by the Retarget Workbench animation pipeline.
IMPORTANT: not a replacement for ``src.core.retargeting.fbx_importer``.
IMPORTANT: keep this quarantined from animation retargeting workflows.
"""

import os, struct, math, logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Measurement-system note: exporters consume raw KotorModel scene values in the
# active system unit. UI display-unit formatting is intentionally not read here.
# Format-specific unit conversion belongs at the exporter boundary only when a
# target format explicitly requires it.
from src.core.geometry.model_data import (
    KotorModel,
    ModelNode,
    NodeFlags,
    GameVersion,
    VertexSkinData,
    BoneWeight,
    _quat_rotate,
    _quat_conjugate,
    _quat_normalize_bind,
    _quat_normalize,
    _quat_mul,
    Animation,
)
from src.io.export_control import TextureSidecarResult, check_export_cancelled

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OBJExportResult:
    """Files and texture-sidecar outcome produced by :class:`OBJExporter`."""

    obj_path: str
    mtl_path: str
    texture_sidecars: TextureSidecarResult
    baked_lightmaps_saved: int = 0


# ──────────────────────────────────────────────────────────────────────
#  OBJ Importer
# ──────────────────────────────────────────────────────────────────────

class OBJImporter:
    def import_obj(self, obj_path: str,
                   model_name: str = "",
                   game_version: GameVersion = GameVersion.K1,
                   supermodel: str = "NULL",
                   classification: str = "character") -> KotorModel:
        """Alias for import_file() – backward compatible."""
        return self.import_file(obj_path, model_name, game_version, supermodel, classification)

    def import_file(self, obj_path: str,
                    model_name: str = "",
                    game_version: GameVersion = GameVersion.K1,
                    supermodel: str = "NULL",
                    classification: str = "character") -> KotorModel:
        p = Path(obj_path)
        if not model_name: model_name = p.stem[:32]

        model = KotorModel(name=model_name, supermodel=supermodel,
                           game_version=game_version, classification=classification)
        cls_map = {'character':2,'tile':1,'door':4,'effect':0}
        model.model_type = cls_map.get(classification.lower(), 2)

        root = ModelNode(name=model_name, flags=int(NodeFlags.HEADER))
        model.root_node = root

        materials: Dict[str, Dict] = {}
        mtl_path = p.with_suffix('.mtl')
        if mtl_path.exists():
            materials = self._parse_mtl(str(mtl_path))

        all_v:  List[Tuple[float,float,float]] = []
        all_vt: List[Tuple[float,float]]       = []
        all_vn: List[Tuple[float,float,float]] = []

        current_mat  = ""
        current_name = "mesh_0"
        groups: List[Dict] = []
        cur: Optional[Dict] = None

        def flush():
            nonlocal cur
            if cur and cur['faces']: groups.append(cur)
            cur = None

        def new_group(n, m=""):
            nonlocal cur, current_name, current_mat
            flush()
            current_name = n[:32]
            current_mat  = m or current_mat
            cur = {'name': current_name, 'mat': current_mat, 'faces': []}

        with open(obj_path, 'r', encoding='utf-8', errors='replace') as f:
            for raw in f:
                ln = raw.strip()
                if not ln or ln[0]=='#': continue
                t  = ln.split()
                c  = t[0].lower()
                if   c=='v'  and len(t)>=4: all_v.append((float(t[1]),float(t[2]),float(t[3])))
                elif c=='vt' and len(t)>=3: all_vt.append((float(t[1]),1.0-float(t[2])))
                elif c=='vn' and len(t)>=4: all_vn.append((float(t[1]),float(t[2]),float(t[3])))
                elif c in ('g','o') and len(t)>1: new_group(t[1])
                elif c=='usemtl' and len(t)>1:
                    current_mat = t[1]
                    if cur: cur['mat'] = current_mat
                    else:   new_group(f"mesh_{len(groups)}", current_mat)
                elif c=='f':
                    if cur is None: new_group(f"mesh_{len(groups)}")
                    parsed = [self._fv(tok) for tok in t[1:]]
                    for i in range(1, len(parsed)-1):
                        cur['faces'].append([parsed[0], parsed[i], parsed[i+1]])
        flush()

        for gi, g in enumerate(groups):
            if not g['faces']: continue
            node = self._make_node(g, all_v, all_vt, all_vn, materials, gi)
            node.parent = root
            root.children.append(node)

        model.compute_bounds()
        return model

    def _fv(self, tok: str) -> Tuple[int,int,int]:
        p = tok.split('/')
        vi  = int(p[0])-1 if p[0]   else 0
        vti = int(p[1])-1 if len(p)>1 and p[1] else -1
        vni = int(p[2])-1 if len(p)>2 and p[2] else -1
        return vi, vti, vni

    def _make_node(self, g, all_v, all_vt, all_vn, materials, idx) -> ModelNode:
        node = ModelNode(name=g['name'],
                         flags=int(NodeFlags.HEADER|NodeFlags.MESH),
                         index=idx)
        mat = materials.get(g['mat'], {})
        node.texture = mat.get('diff_map', g['mat'])[:32] if g['mat'] else ""
        node.diffuse = mat.get('diffuse', (0.8,0.8,0.8))
        node.ambient = mat.get('ambient', (0.2,0.2,0.2))
        # Mark as explicitly imported (not parsed from MDL) so the viewport
        # renderer does not misclassify it as a KotOR deformation-helper.
        node.render    = True
        node._imported = True

        idx_map: Dict[Tuple,int] = {}
        ov: List[Tuple[float,float,float]] = []
        out: List[Tuple[float,float]]       = []
        on: List[Tuple[float,float,float]] = []
        of: List[Tuple[int,int,int]]        = []

        def gi_(vi,vti,vni):
            key = (vi,vti,vni)
            if key in idx_map: return idx_map[key]
            n = len(ov)
            idx_map[key] = n
            ov.append(all_v[vi] if 0<=vi<len(all_v) else (0,0,0))
            out.append(all_vt[vti] if 0<=vti<len(all_vt) else (0,0))
            on.append(all_vn[vni] if 0<=vni<len(all_vn) else (0,0,1))
            return n

        for tri in g['faces']:
            a,b,c = gi_(*tri[0]), gi_(*tri[1]), gi_(*tri[2])
            of.append((a,b,c))

        node.vertices = ov; node.uvs = out
        node.normals  = on if any(x!=(0,0,1) for x in on) else self._flat_normals(ov,of)
        node.faces    = of
        node.compute_bounds()
        return node

    def _flat_normals(self, verts, faces):
        ns = [[0.0,0.0,0.0]] * len(verts)
        cnt= [0] * len(verts)
        for v1,v2,v3 in faces:
            if max(v1,v2,v3) >= len(verts): continue
            ax,ay,az=verts[v1]; bx,by,bz=verts[v2]; cx,cy,cz=verts[v3]
            ux,uy,uz=bx-ax,by-ay,bz-az; vx,vy,vz=cx-ax,cy-ay,cz-az
            nx=uy*vz-uz*vy; ny=uz*vx-ux*vz; nz=ux*vy-uy*vx
            l=math.sqrt(nx*nx+ny*ny+nz*nz) or 1
            nx/=l;ny/=l;nz/=l
            for vi in (v1,v2,v3):
                ns[vi][0]+=nx; ns[vi][1]+=ny; ns[vi][2]+=nz; cnt[vi]+=1
        result = []
        for i,n in enumerate(ns):
            if cnt[i]: l=math.sqrt(sum(x*x for x in n)) or 1; result.append((n[0]/l,n[1]/l,n[2]/l))
            else: result.append((0,0,1))
        return result

    def _parse_mtl(self, path: str) -> Dict[str,Dict]:
        mats: Dict[str,Dict] = {}; cur = None
        with open(path,'r',encoding='utf-8',errors='replace') as f:
            for ln in f:
                t = ln.strip().split()
                if not t: continue
                c = t[0].lower()
                if c=='newmtl' and len(t)>1: cur=t[1]; mats[cur]={}
                elif cur:
                    if   c=='kd' and len(t)>=4: mats[cur]['diffuse']=(float(t[1]),float(t[2]),float(t[3]))
                    elif c=='ka' and len(t)>=4: mats[cur]['ambient']=(float(t[1]),float(t[2]),float(t[3]))
                    elif c=='map_kd' and len(t)>1: mats[cur]['diff_map']=Path(t[-1]).stem[:32]
                    elif c=='map_bump' and len(t)>1: mats[cur]['bump_map']=Path(t[-1]).stem[:32]
        return mats


# ──────────────────────────────────────────────────────────────────────
#  FBX Importer (via pyassimp / assimp_py / trimesh)
# ──────────────────────────────────────────────────────────────────────
# Import priority:
#   1. pyassimp  – full bone/skin support, requires native Assimp DLL/so
#   2. assimp_py – bundled native lib, no bone data (geometry-only fallback)
#   3. trimesh   – pure Python, handles FBX ASCII + OBJ/GLB
# ──────────────────────────────────────────────────────────────────────

class FBXImporter:
    def import_file(self, path: str,
                    model_name: str = "",
                    game_version: GameVersion = GameVersion.K1,
                    supermodel: str = "NULL",
                    classification: str = "character") -> Optional[KotorModel]:
        if not model_name: model_name = Path(path).stem[:32]
        # 1) pyassimp (full bone/skin support, needs native DLL)
        try:
            return self._load_assimp(path, model_name, game_version, supermodel, classification)
        except ImportError:
            log.debug("pyassimp not available for FBX import — trying assimp_py")
        except BaseException as e:
            # BaseException: AssimpError extends BaseException, not Exception
            log.debug(f"pyassimp FBX load failed: {e} — trying assimp_py")
        # 2) assimp_py (bundled native lib, geometry only — no bones)
        try:
            result = self._load_assimp_py(path, model_name, game_version, supermodel, classification)
            if result is not None:
                log.info("FBX imported via assimp_py (geometry only — no bone/skin data)")
                return result
        except ImportError:
            log.debug("assimp_py not available — trying trimesh")
        except Exception as e:
            log.debug(f"assimp_py FBX load failed: {e} — trying trimesh")
        # 3) trimesh (pure Python fallback)
        try:
            return self._load_trimesh(path, model_name, game_version, supermodel, classification)
        except ImportError:
            log.debug("trimesh not available for FBX import")
        except Exception as e:
            log.debug(f"trimesh FBX load failed: {e}")
        # 4) Blender headless fallback for binary UE/FBX files that Assimp/trimesh
        # cannot read in the current Python environment.
        try:
            from src.converters.blender_fbx_mesh_importer import import_fbx_mesh_with_blender

            return import_fbx_mesh_with_blender(
                path,
                model_name=model_name,
                game_version=game_version,
                supermodel=supermodel,
                classification=classification,
            )
        except ImportError:
            log.debug("Blender FBX mesh fallback importer not available")
        except Exception as e:
            log.debug(f"Blender FBX mesh fallback failed: {e}")
        log.error(
            "FBX import failed — no suitable library found.\n"
            "  Option A: pip install pyassimp  (+ Assimp DLL for bone/skin data)\n"
            "  Option B: pip install assimp-py (bundled DLL, geometry only)\n"
            "  Option C: pip install trimesh   (pure Python, limited FBX support)\n"
            "  Option D: install Blender 4.2 for GhostRigger's headless mesh preview fallback\n"
            "  Or export your model as OBJ or GLB for import."
        )
        return None

    def _load_assimp(self, path, model_name, gv, sm, cl) -> KotorModel:
        import pyassimp, pyassimp.postprocess as pp
        flags = (pp.aiProcess_Triangulate | pp.aiProcess_GenSmoothNormals |
                 pp.aiProcess_JoinIdenticalVertices | pp.aiProcess_LimitBoneWeights |
                 pp.aiProcess_CalcTangentSpace)
        scene = pyassimp.load(path, processing=flags)
        model = KotorModel(name=model_name, supermodel=sm, game_version=gv, classification=cl)
        root  = ModelNode(name=model_name, flags=int(NodeFlags.HEADER))
        model.root_node = root

        # Skeleton: walk assimp node tree for bone hierarchy
        self._walk_assimp_nodes(scene.rootnode, root, scene)

        pyassimp.release(scene)
        model.compute_bounds()
        return model

    # ── assimp_py fallback (bundled native lib, no bone data) ─────────
    def _load_assimp_py(self, path, model_name, gv, sm, cl) -> KotorModel:
        """Import via assimp_py — geometry, normals, UVs, materials.
        Does NOT import bone/skin data (assimp_py doesn't expose it)."""
        import assimp_py
        flags = (assimp_py.Process_Triangulate |
                 assimp_py.Process_GenSmoothNormals |
                 assimp_py.Process_JoinIdenticalVertices |
                 assimp_py.Process_CalcTangentSpace)
        scene = assimp_py.import_file(str(path), flags)
        model = KotorModel(name=model_name, supermodel=sm, game_version=gv, classification=cl)
        root  = ModelNode(name=model_name, flags=int(NodeFlags.HEADER))
        model.root_node = root

        # Walk node tree
        self._walk_assimp_py_nodes(scene.root_node, root, scene)

        model.compute_bounds()
        return model

    def _walk_assimp_py_nodes(self, ai_node, parent_node, scene):
        """Recursively mirror assimp_py scene graph into KotorModel nodes."""
        mat = ai_node.transformation  # 4x4 row-major flat or nested
        # assimp_py returns transformation as a 4x4 list-of-lists
        if isinstance(mat, (list, tuple)) and len(mat) == 4:
            tx, ty, tz = float(mat[0][3]), float(mat[1][3]), float(mat[2][3])
        else:
            tx = ty = tz = 0.0

        node = ModelNode(name=(ai_node.name or "node")[:32],
                         flags=int(NodeFlags.HEADER),
                         position=(tx, ty, tz),
                         parent=parent_node)
        parent_node.children.append(node)

        # Meshes attached to this node
        for mesh_idx in (ai_node.mesh_indices or []):
            if mesh_idx < len(scene.meshes):
                mesh = scene.meshes[mesh_idx]
                mesh_node = self._assimp_py_mesh(mesh, scene)
                mesh_node.parent = node
                node.children.append(mesh_node)

        for child in (ai_node.children or []):
            self._walk_assimp_py_nodes(child, node, scene)

    def _assimp_py_mesh(self, ai_mesh, scene) -> ModelNode:
        """Convert an assimp_py Mesh to a ModelNode (geometry only)."""
        flags = int(NodeFlags.HEADER | NodeFlags.MESH)
        node = ModelNode(name=(ai_mesh.name or "mesh")[:32], flags=flags)

        # Vertices (flat list of floats, stride 3)
        verts = ai_mesh.vertices
        node.vertices = [(verts[i], verts[i+1], verts[i+2])
                         for i in range(0, len(verts), 3)]

        # Normals
        norms = ai_mesh.normals
        if norms:
            node.normals = [(norms[i], norms[i+1], norms[i+2])
                            for i in range(0, len(norms), 3)]

        # UVs — first channel (texcoords is list of channels, each flat)
        if ai_mesh.texcoords and len(ai_mesh.texcoords) > 0:
            uv_ch = ai_mesh.texcoords[0]
            n_comp = ai_mesh.num_uv_components[0] if ai_mesh.num_uv_components else 2
            node.uvs = [(uv_ch[i], 1.0 - uv_ch[i+1])
                        for i in range(0, len(uv_ch), n_comp)]

        # Faces (flat index list, all triangulated)
        idx = ai_mesh.indices
        node.faces = [(idx[i], idx[i+1], idx[i+2])
                      for i in range(0, len(idx), 3)]

        # Material — extract diffuse texture name
        if scene.materials and ai_mesh.material_index < len(scene.materials):
            mat = scene.materials[ai_mesh.material_index]
            # assimp_py materials: dict of (key, semantic, index) -> value
            if isinstance(mat, dict):
                for k, v in mat.items():
                    if isinstance(k, tuple) and len(k) >= 2:
                        if k[1] == 1 and k[0] == '$tex.file':  # diffuse texture
                            node.texture = Path(str(v)).stem[:32]
                        elif k[0] == '$clr.diffuse' and isinstance(v, (list, tuple)):
                            node.diffuse = (float(v[0]), float(v[1]), float(v[2]))

        node.render    = True
        node._imported = True
        node.compute_bounds()
        return node

    def _walk_assimp_nodes(self, ai_node, parent_node, scene):
        """Recursively mirror the assimp scene graph into KotorModel nodes"""
        import numpy as np
        mat = ai_node.transformation
        # Extract translation from 4x4 matrix (row-major in assimp)
        tx, ty, tz = float(mat[0][3]), float(mat[1][3]), float(mat[2][3])

        node = ModelNode(name=ai_node.name[:32] or "node",
                         flags=int(NodeFlags.HEADER),
                         position=(tx, ty, tz),
                         parent=parent_node)
        parent_node.children.append(node)

        # Check if this node has associated mesh
        for mesh_idx in ai_node.meshes:
            mesh = scene.meshes[mesh_idx]
            mesh_node = self._assimp_mesh(mesh, scene)
            mesh_node.parent = node
            node.children.append(mesh_node)

        for child in ai_node.children:
            self._walk_assimp_nodes(child, node, scene)

    def _assimp_mesh(self, ai_mesh, scene) -> ModelNode:
        is_skin = hasattr(ai_mesh,'bones') and len(ai_mesh.bones)>0
        flags   = int(NodeFlags.HEADER|NodeFlags.MESH)
        if is_skin: flags |= int(NodeFlags.SKIN)

        node = ModelNode(name=ai_mesh.name[:32] or "mesh", flags=flags)
        node.vertices = [(float(v[0]),float(v[1]),float(v[2])) for v in ai_mesh.vertices]
        if len(ai_mesh.normals):
            node.normals = [(float(n[0]),float(n[1]),float(n[2])) for n in ai_mesh.normals]
        if len(ai_mesh.texturecoords) and len(ai_mesh.texturecoords[0]):
            node.uvs = [(float(u[0]),1.0-float(u[1])) for u in ai_mesh.texturecoords[0]]
        node.faces = [(int(f.indices[0]),int(f.indices[1]),int(f.indices[2])) for f in ai_mesh.faces]

        # Material
        if scene.materials and ai_mesh.materialindex < len(scene.materials):
            mat = scene.materials[ai_mesh.materialindex]
            for prop in mat.properties:
                k = prop.key.lower()
                if '$tex.file' in k and '1,0' in k:
                    node.texture = Path(str(prop.data)).stem[:32]
                elif '$clr.diffuse' in k:
                    d = prop.data
                    node.diffuse = (float(d[0]),float(d[1]),float(d[2]))

        # Skin
        if is_skin:
            n_verts = len(node.vertices)
            wt = [[BoneWeight(0,0.0)]*4 for _ in range(n_verts)]
            sc = [0]*n_verts
            node.bone_map = []
            for bi, bone in enumerate(ai_mesh.bones):
                node.bone_map.append(bone.name[:32])
                for w in bone.weights:
                    vi = w.vertexid
                    if vi < n_verts and sc[vi] < 4:
                        wt[vi][sc[vi]] = BoneWeight(bi, w.weight); sc[vi]+=1
            node.skin_data = [VertexSkinData(influences=[b for b in row if b.weight>0]) for row in wt]

        # Mark as imported so the viewport renderer skips deformation-helper filtering
        node.render    = True
        node._imported = True
        node.compute_bounds()
        return node

    def _load_trimesh(self, path, model_name, gv, sm, cl) -> KotorModel:
        import trimesh
        scene = trimesh.load(path)
        model = KotorModel(name=model_name, supermodel=sm, game_version=gv, classification=cl)
        root  = ModelNode(name=model_name, flags=int(NodeFlags.HEADER))
        model.root_node = root
        geoms = scene.geometry if hasattr(scene,'geometry') else {model_name:scene}
        for gname, mesh in geoms.items():
            n = ModelNode(name=gname[:32], flags=int(NodeFlags.HEADER|NodeFlags.MESH), parent=root)
            n.vertices = [tuple(v) for v in mesh.vertices.tolist()]
            n.faces    = [tuple(f) for f in mesh.faces.tolist()]
            if hasattr(mesh,'vertex_normals') and mesh.vertex_normals is not None:
                n.normals = [tuple(x) for x in mesh.vertex_normals.tolist()]
            if hasattr(mesh,'visual') and hasattr(mesh.visual,'uv') and mesh.visual.uv is not None:
                n.uvs = [(float(u),1.0-float(v)) for u,v in mesh.visual.uv.tolist()]
            # Mark as imported so the viewport renderer skips deformation-helper filtering
            n.render    = True
            n._imported = True
            n.compute_bounds()
            root.children.append(n)
        model.compute_bounds()
        return model


# ──────────────────────────────────────────────────────────────────────
#  KotorModel → OBJ Exporter
# ──────────────────────────────────────────────────────────────────────

class OBJExporter:
    """
    Export KotorModel to OBJ + MTL.

    KotOR UV convention: V is stored bottom-up (OpenGL style, 0=bottom, 1=top).
    OBJ format stores V the same way but Blender displays UVs with Y-up, so we
    flip V on export: vt_out = 1.0 - v_kotor.

    Only *renderable* nodes are exported:
      - Nodes with ``render=False`` are invisible engine-internal nodes (collision
        proxies, occluders, camera helpers) and are never exported.
      - Deformation-helper trimeshes (texture='null', _g/_dum suffix)
        are skipped; they are used by the engine's SkinMesh pipeline and do not
        represent visible geometry.
    """

    # ── Helpers that mirror the viewport's _is_deformation_helper logic ──────

    @staticmethod
    def _clean_tex(name: str) -> str:
        """Return printable ASCII tex name, stripped."""
        if not name:
            return ''
        cleaned = ''.join(c for c in name if 32 <= ord(c) < 127)
        return cleaned.strip()

    # ── Facial geometry node names (always renderable regardless of flags) ──
    # KotOR head MDLs contain eye, eyelid, teeth, tongue, gum and jaw meshes
    # as visible geometry.  Some models store render=0 or extreme/absent UVs on
    # these nodes, which would cause the generic helper-filter to incorrectly
    # discard them.  We match them by SUBSTRING (not prefix only) to cover both
    # standard PC head naming (eyeRA, eyeLA, teethlower, teethupper, tongue) and
    # NPC naming conventions (f_rlweye_g, f_llweye_g, f_teetha_g, etc.).
    #
    # Mirrors _INNER_GEO_SUBSTRINGS in viewport.py::FrameRenderer.
    _FACIAL_MESH_PREFIXES: tuple = (
        # Eyeballs (left/right, regular + specular layer)
        'lseyeball', 'rseyeball', 'lssupeyeball', 'rssupeyeball',
        # Teeth and tongue — prefix matches cover most names
        'teethlower', 'teethupper', 'teeth',
        'tongue',
        # Eyelids (some models have separate eyelid meshes)
        'eyelidl', 'eyelidr', 'eyelid_l', 'eyelid_r',
        # Eye whites / cornea layers
        'eyewhite', 'eyecornea',
    )

    # Substring set that mirrors viewport._INNER_GEO_SUBSTRINGS.
    # A node whose name CONTAINS any of these strings is treated as facial
    # geometry regardless of prefix, suffix, or render flag.
    _FACIAL_MESH_SUBSTRINGS: tuple = (
        'eye', 'lid', 'teeth', 'tooth', 'gum', 'jaw',
        'tongue', 'teethu', 'teethl',
    )

    @classmethod
    def _is_facial_geometry(cls, node) -> bool:
        """
        Return True if this node is KotOR facial geometry (eyes, teeth, tongue,
        eyelids, gums, jaw) that must be included in exports regardless of its
        render flag or UV coordinates.

        Matching strategy (mirrors viewport._is_deformation_helper v26 logic):
          1. Prefix match against _FACIAL_MESH_PREFIXES  (fast, covers standard PC names)
          2. Substring match against _FACIAL_MESH_SUBSTRINGS  (covers NPC names such as
             f_rlweye_g, f_llweye_g, f_teetha_g, jawbone_g, etc.)

        A real texture AND UVs are additionally required so
        that bare bone-helper dummies accidentally named 'jaw_g' are NOT promoted.
        The vertex-count guard in _is_renderable acts as a final safety net.
        """
        name_lower = getattr(node, 'name', '').lower()
        # Fast path: exact prefix match (standard K1/K2 PC head node names)
        if any(name_lower.startswith(p) for p in cls._FACIAL_MESH_PREFIXES):
            return True
        # Substring match: covers NPC head nodes like f_rlweye_g, f_llweye_g,
        # f_teetha_g, jawskin, gumskin, tonguemesh, etc.
        if any(s in name_lower for s in cls._FACIAL_MESH_SUBSTRINGS):
            # Require a real texture + valid UVs so skeletal bone-helpers
            # that happen to contain 'jaw' or 'gum' in their names are not promoted.
            tex = cls._clean_tex(getattr(node, 'texture', '') or '')
            if tex and tex.upper() != 'NULL':
                uvs = getattr(node, 'uvs', []) or []
                if uvs:
                    return True
            # Even without a texture: if the node has vertices and is explicitly
            # named with a strong facial keyword, treat it as facial geometry.
            # This handles nodes like 'teethUpper' that carry UVs but whose
            # texture name resolves at runtime from the head's multi-texture.
            if any(name_lower.startswith(p)
                   for p in ('teeth', 'tongue', 'eyelid', 'eyeball')):
                return True
        return False

    @classmethod
    def _is_deformation_helper(cls, node) -> bool:
        """
        Return True if this mesh node is a hidden deformation-helper that
        should NOT appear in the exported geometry.  Mirrors the logic in
        viewport.py::FrameRenderer._is_deformation_helper.

        Facial geometry nodes (eyes, teeth, tongue, gums, jaw) are NEVER helpers
            even if they carry large tiled UVs or a null texture — some NPC head variants
        store these with render=0 or broken UV coordinates in the binary MDL.
        """
        # Facial geometry is always visible — never classify as a helper
        if cls._is_facial_geometry(node):
            return False

        tex = cls._clean_tex(getattr(node, 'texture', '') or '')
        is_null_tex = (not tex or tex.upper() == 'NULL')
        is_skin = getattr(node, 'is_skin', False)
        uvs = getattr(node, 'uvs', []) or []

        # Skin node with a real texture and UVs -> visible
        if is_skin and not is_null_tex and uvs:
            return False

        # Non-skin _g / _G / _dum nodes → always helpers
        # EXCEPTION: inner-geometry nodes whose names CONTAIN facial substrings
        # and carry a real texture ARE renderable (e.g. f_rlweye_g NPC eyeballs).
        name_lower = getattr(node, 'name', '').lower()
        _name_is_inner_geo = any(s in name_lower for s in cls._FACIAL_MESH_SUBSTRINGS)
        if not is_skin and (name_lower.endswith('_g')
                            or name_lower.endswith('_g0')
                            or name_lower.endswith('_dum')):
            # Inner-geo NPC eyeball/teeth nodes (e.g. f_rlweye_g) with a real
            # texture and UVs are renderable despite the _g suffix.
            if _name_is_inner_geo and not is_null_tex and uvs:
                return False
            return True

        # Null-texture non-skin nodes → helpers
        if is_null_tex and not is_skin:
            return True

        # Null-texture skin nodes with no/zero UVs → helpers
        if is_null_tex and is_skin and (not uvs
                or all(u == 0.0 and v == 0.0 for u, v in uvs[:5])):
            return True

        return False

    @classmethod
    def _is_renderable(cls, node) -> bool:
        """
        Return True if the node should be included in the exported geometry.

        Criteria:
          1. Has vertices.
          2. render flag is not explicitly False — UNLESS the node is known
             facial geometry (eyes/teeth/tongue/jaw/gum/eyelids), which some
             KotOR NPC heads incorrectly store with render=0 in the binary MDL.
          3. Not a deformation helper.
          4. Not an emitter or light node.
        """
        if bool(getattr(node, '_gr_bas_geometry_replaced_by_attachment', False)):
            return False
        if not getattr(node, 'vertices', None):
            return False
        # Facial geometry nodes bypass the render=False gate entirely
        is_facial = cls._is_facial_geometry(node)
        if not is_facial and not getattr(node, 'render', True):
            return False
        if getattr(node, 'is_emitter', False) or getattr(node, 'is_light', False):
            return False
        # Level assembly has already selected real room/placement surfaces.
        # Untextured blockout meshes are valid geometry, not rig helpers.
        if bool(getattr(node, '_gr_level_export_geometry', False)):
            return True
        if cls._is_deformation_helper(node):
            return False
        return True

    # ── Bind-pose world-space transform helpers ──────────────────────────────

    @staticmethod
    def _node_bind_world_transform(node):
        """Compose the emitted object hierarchy with standard rigid math."""

        local_position = tuple(float(value) for value in getattr(node, "position", (0.0, 0.0, 0.0)))
        local_rotation = _quat_normalize(
            tuple(float(value) for value in getattr(node, "rotation", (0.0, 0.0, 0.0, 1.0)))
        )
        parent = getattr(node, "parent", None)
        if parent is None:
            return local_position, local_rotation
        parent_position, parent_rotation = OBJExporter._node_bind_world_transform(parent)
        rotated_position = _quat_rotate(parent_rotation, local_position)
        world_position = tuple(
            parent_position[index] + rotated_position[index]
            for index in range(3)
        )
        world_rotation = _quat_normalize(_quat_mul(parent_rotation, local_rotation))
        return world_position, world_rotation

    @staticmethod
    def _node_bind_world_verts(node) -> List[Tuple[float, float, float]]:
        """
        Return vertices transformed to bind-pose world space.

        For skin nodes (is_skin=True) the vertices are stored in skin-node-local
        space.  The bind-pose world position is:

            v_world = rotate(skin_wo, v_local) + skin_wp

        where (skin_wp, skin_wo) is composed from the emitted local hierarchy.
        When the skin-node rotation is the
        identity quaternion (the common case) this simplifies to:

            v_world = v_local + skin_wp

        For non-skin trimesh nodes the same formula applies, but the node's own
        rotation is *not* collapsed (see model_data.world_transform leaf-node
        handling), so the full rotation is applied to vertex positions.

        NOTE: This does NOT apply Linear Blend Skinning — it produces the
        correct bind-pose shape only for nodes whose vertex data is already
        stored in world/model space (simple non-skin meshes) OR for skin nodes
        whose raw vertex data represents bind-pose local coords that need only
        the skin node's world translation applied.  For c_bosdrexl and similar
        creatures the skin vertices ARE in node-local space and their correct
        placement requires this world-space offset.
        """
        verts = node.vertices or []
        if not verts:
            return []

        wp, wo = OBJExporter._node_bind_world_transform(node)
        wo_rot = math.sqrt(wo[0]**2 + wo[1]**2 + wo[2]**2)
        is_id = wo_rot < 0.001

        if is_id:
            return [(v[0] + wp[0], v[1] + wp[1], v[2] + wp[2]) for v in verts]
        else:
            result = []
            for v in verts:
                rv = _quat_rotate(wo, v)
                result.append((rv[0] + wp[0], rv[1] + wp[1], rv[2] + wp[2]))
            return result

    @staticmethod
    def _node_bind_world_normals(node) -> List[Tuple[float, float, float]]:
        """
        Return normals rotated by the node's bind-pose world rotation.
        Normals are direction vectors — translate only applies to positions, not
        directions, so only the rotation component is applied.
        """
        normals = node.normals or []
        if not normals:
            return []

        _wp, wo = OBJExporter._node_bind_world_transform(node)
        wo_rot = math.sqrt(wo[0]**2 + wo[1]**2 + wo[2]**2)
        is_id = wo_rot < 0.001

        if is_id:
            return list(normals)   # no rotation to apply
        else:
            return [_quat_rotate(wo, n) for n in normals]

    def export(self, model: KotorModel, obj_path: str, tex_cache=None,
               export_rigging: bool = True, is_cancelled=None) -> OBJExportResult:
        """
        Export model to OBJ + MTL.

        Parameters
        ----------
        model          : KotorModel to export.
        obj_path       : Output .obj file path.
        tex_cache      : Optional TextureCache object.  When provided, textures are
                         looked up and saved as TGA files next to the OBJ so that the
                         MTL map_Kd references resolve correctly in Blender/Maya.
        export_rigging : When True (default) a 'rigging/' subdirectory is created
                         next to the OBJ file and receives:
                           rigging/<model>.skeleton.json  – full bone hierarchy
                           rigging/<model>.<anim>.json    – one file per animation
                           rigging/<model>.weights.json   – per-vertex skin weights
                         This data is needed by tools (e.g. Maya scripts) to
                         re-attach rigging after import.
        """
        p = Path(obj_path); mp = p.with_suffix('.mtl')
        obj_lines = [f"# GhostRigger-K1-K2 export – {model.name}", f"mtllib {mp.name}", ""]
        mtl_lines = ["# GhostRigger-K1-K2 materials", ""]
        material_names_by_key: dict[tuple, str] = {}
        used_material_names: set[str] = set()
        out_dir = p.parent

        # Count skipped nodes for the header comment
        all_mesh  = list(model.mesh_nodes())
        renderable = [n for n in all_mesh if self._is_renderable(n)]
        skipped_render = sum(1 for n in all_mesh if not getattr(n, 'render', True))
        skipped_helper = sum(1 for n in all_mesh if getattr(n, 'render', True)
                             and self._is_deformation_helper(n)
                             and getattr(n, 'vertices', None))
        obj_lines.insert(1,
            f"# mesh nodes total={len(all_mesh)}  "
            f"exported={len(renderable)}  "
            f"skipped_render_false={skipped_render}  "
            f"skipped_deform_helpers={skipped_helper}")

        vo = vto = vno = 0
        for node in renderable:
            check_export_cancelled(is_cancelled)
            # Apply bind-pose world transform so vertices are in model/world space.
            # This is essential for skin nodes: KotOR stores skin vertices in
            # node-local space; without this offset, body-segment nodes like
            # Rwing_06 on c_bosdrexl appear displaced far from the creature body.
            world_verts   = self._node_bind_world_verts(node)
            world_normals = self._node_bind_world_normals(node)

            nv  = len(world_verts)
            nuv = len(node.uvs)
            nno = len(world_normals)

            obj_lines.append(f"o {node.name}")
            obj_lines.append(f"# verts={nv} uvs={nuv} normals={nno} faces={len(node.faces)}")

            for vertex_index, (x, y, z) in enumerate(world_verts):
                if vertex_index % 1024 == 0:
                    check_export_cancelled(is_cancelled)
                obj_lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")

            # Write UVs – KotOR stores V bottom-up, OBJ convention is same.
            # We flip V here so that the exported file looks correct in Blender
            # (which loads UVs as-is and uses OpenGL convention Y-up).
            for uv_index, (u,v) in enumerate(node.uvs):
                if uv_index % 1024 == 0:
                    check_export_cancelled(is_cancelled)
                obj_lines.append(f"vt {u:.6f} {(1.0 - v):.6f}")

            for normal_index, (x, y, z) in enumerate(world_normals):
                if normal_index % 1024 == 0:
                    check_export_cancelled(is_cancelled)
                obj_lines.append(f"vn {x:.6f} {y:.6f} {z:.6f}")

            # A merged scene mesh may carry multiple material slots plus one
            # face-material index per polygon.  Ordinary models retain the
            # original single-node material behavior through the fallback.
            material_slots = list(
                getattr(node, '_gr_fbx_material_slots', None) or [node]
            )
            material_names = []
            for material_index, material_node in enumerate(material_slots):
                check_export_cancelled(is_cancelled)
                raw_tex = (
                    getattr(material_node, 'texture_clean', '')
                    or getattr(material_node, 'texture', '')
                    or ''
                )
                tex_name = self._clean_tex(raw_tex)
                if tex_name.upper() in ('NULL', 'BLACK', ''):
                    tex_name = ''
                material_key = (
                    tex_name.casefold(),
                    tuple(float(value) for value in material_node.diffuse),
                    tuple(float(value) for value in material_node.ambient),
                    tuple(float(value) for value in material_node.specular),
                    float(material_node.shininess),
                    float(material_node.alpha),
                )
                mat_name = material_names_by_key.get(material_key)
                if mat_name is not None:
                    material_names.append(mat_name)
                    continue
                base_name = tex_name if tex_name else (
                    getattr(material_node, 'name', '')
                    or f"{node.name}_material_{material_index + 1}"
                )
                mat_name = base_name
                suffix = 2
                while mat_name.casefold() in used_material_names:
                    mat_name = f"{base_name}_{suffix}"
                    suffix += 1
                material_names_by_key[material_key] = mat_name
                used_material_names.add(mat_name.casefold())
                material_names.append(mat_name)
                r,g,b = material_node.diffuse
                ar,ag,ab = material_node.ambient
                sr,sg,sb = material_node.specular
                mtl_lines += [
                    f"newmtl {mat_name}",
                    f"Ka {ar:.4f} {ag:.4f} {ab:.4f}",
                    f"Kd {r:.4f} {g:.4f} {b:.4f}",
                    f"Ks {sr:.4f} {sg:.4f} {sb:.4f}",
                    f"Ns {max(material_node.shininess, 0.0):.2f}",
                    f"d  {material_node.alpha:.4f}",
                ]
                if tex_name:
                    mtl_lines.append(f"map_Kd {tex_name}.tga")
                mtl_lines.append("")

            # Faces: emit f v/vt/vn triples; clamp UV and normal indices safely
            huv = (nuv > 0)
            hn  = (nno > 0)
            obj_lines.append("s 1")  # smoothing group

            # Snapshot current global offsets to avoid closure capture issues
            _vo, _vto, _vno = vo, vto, vno

            # KotOR ASCII MDL models use SEPARATE UV indices (tvert indices) stored
            # in node.face_uvs, which differ from the vertex position indices.
            # Binary MDL models have per-vertex UVs where UV index == vertex index.
            # BUG FIX v20 (ebon_01 K2 Maya texture fix):
            #   Previously, face UV indices were always taken from the vertex position
            #   index (v1/v2/v3), which is WRONG for ASCII-format models.  This caused
            #   completely scrambled UV mapping in Maya / Blender for any ASCII-sourced
            #   model, manifesting as stripe-pattern distortions across the hull.
            # Fix: when face_uvs is present, use the per-face tvert index tuple instead.
            face_uvs_data = getattr(node, 'face_uvs', []) or []
            _has_face_uvs = bool(face_uvs_data) and len(face_uvs_data) == len(node.faces)

            face_materials = list(getattr(node, 'face_mats', None) or [])
            active_material = None
            for fi, (v1, v2, v3) in enumerate(node.faces):
                if fi % 512 == 0:
                    check_export_cancelled(is_cancelled)
                if max(v1, v2, v3) >= nv:
                    continue   # skip degenerate faces with out-of-range indices

                material_index = int(face_materials[fi]) if fi < len(face_materials) else 0
                material_index = max(0, min(material_index, len(material_names) - 1))
                mat_name = material_names[material_index]
                if mat_name != active_material:
                    obj_lines.append(f"usemtl {mat_name}")
                    active_material = mat_name

                # Determine UV indices: use face_uvs tvert indices when present,
                # otherwise fall back to vertex indices (binary MDL convention).
                if _has_face_uvs:
                    fu = face_uvs_data[fi]
                    ui1 = min(fu[0], nuv-1) if nuv > 0 else 0
                    ui2 = min(fu[1], nuv-1) if nuv > 0 else 0
                    ui3 = min(fu[2], nuv-1) if nuv > 0 else 0
                else:
                    ui1 = min(v1, nuv-1) if nuv > 0 else 0
                    ui2 = min(v2, nuv-1) if nuv > 0 else 0
                    ui3 = min(v3, nuv-1) if nuv > 0 else 0

                ni1 = min(v1, nno-1) if nno > 0 else 0
                ni2 = min(v2, nno-1) if nno > 0 else 0
                ni3 = min(v3, nno-1) if nno > 0 else 0

                if huv and hn:
                    obj_lines.append(
                        f"f {v1+1+_vo}/{ui1+1+_vto}/{ni1+1+_vno} "
                        f"{v2+1+_vo}/{ui2+1+_vto}/{ni2+1+_vno} "
                        f"{v3+1+_vo}/{ui3+1+_vto}/{ni3+1+_vno}")
                elif huv:
                    obj_lines.append(
                        f"f {v1+1+_vo}/{ui1+1+_vto} "
                        f"{v2+1+_vo}/{ui2+1+_vto} "
                        f"{v3+1+_vo}/{ui3+1+_vto}")
                elif hn:
                    obj_lines.append(
                        f"f {v1+1+_vo}//{ni1+1+_vno} "
                        f"{v2+1+_vo}//{ni2+1+_vno} "
                        f"{v3+1+_vo}//{ni3+1+_vno}")
                else:
                    obj_lines.append(
                        f"f {v1+1+_vo} {v2+1+_vo} {v3+1+_vo}")

            obj_lines.append("")
            vo  += nv
            vto += nuv
            vno += nno

        check_export_cancelled(is_cancelled)
        Path(obj_path).write_text('\n'.join(obj_lines), encoding='utf-8')
        mp.write_text('\n'.join(mtl_lines), encoding='utf-8')
        log.info(f"Exported OBJ → {obj_path}  "
                 f"({vo} verts, {len(renderable)}/{len(all_mesh)} mesh nodes exported)")

        texture_sidecars = self._export_textures_to_dir(
            model,
            out_dir,
            tex_cache,
            is_cancelled=is_cancelled,
        )
        baked_lightmaps_saved = self._export_baked_lightmaps_to_dir(model, out_dir)

        # Export rigging + animations into a dedicated subfolder
        if export_rigging:
            rig_count = _export_rigging_data(model, out_dir)
            if rig_count > 0:
                log.info(f"Rigging data exported: {rig_count} file(s) → {out_dir / 'rigging'}")

        return OBJExportResult(
            obj_path=str(p),
            mtl_path=str(mp),
            texture_sidecars=texture_sidecars,
            baked_lightmaps_saved=baked_lightmaps_saved,
        )

    @staticmethod
    def _export_textures_to_dir(
        model: KotorModel,
        out_dir: Path,
        tex_cache,
        *,
        is_cancelled=None,
    ) -> TextureSidecarResult:
        """
        Save all textures referenced by *model* as TGA files into *out_dir*.

        Cache misses and non-fatal write failures are returned explicitly so
        callers never have to claim sidecars that were not produced.
        """
        requested: list[str] = []
        saved_files: list[str] = []
        missing: list[str] = []
        failed: list[str] = []
        seen: set = set()
        for node in model.mesh_nodes():
            check_export_cancelled(is_cancelled)
            material_slots = list(
                getattr(node, '_gr_fbx_material_slots', None) or [node]
            )
            for material_node in material_slots:
                check_export_cancelled(is_cancelled)
                raw = (
                    getattr(material_node, 'texture_clean', '')
                    or getattr(material_node, 'texture', '')
                    or ''
                )
                tex_name = raw.strip()
                if not tex_name or tex_name.upper() in ('NULL', 'BLACK', ''):
                    continue
                if tex_name.lower() in seen:
                    continue
                seen.add(tex_name.lower())
                requested.append(tex_name)
                if tex_cache is None:
                    missing.append(tex_name)
                    continue
                try:
                    img = tex_cache.get(tex_name)
                    if img is None:
                        missing.append(tex_name)
                        continue
                    out_path = out_dir / f"{tex_name}.tga"
                    if not out_path.exists():
                        img_rgb = img.convert('RGB') if img.mode not in ('RGB', 'RGBA') else img
                        # T2554: decoded KOTOR textures are in game/D3D row order
                        # (the viewport compensates in-shader with 1-v).  The OBJ
                        # exporter writes DCC-convention vt (1-v), so the sidecar
                        # image must be row-flipped for Blender/Maya — same parity
                        # rule as the FBX sidecar path below and the game-import
                        # direction (T2552).  Without this the texture reads
                        # coherently-but-misplaced (V-mirrored atlas islands).
                        try:
                            from PIL import Image as _PILImage
                            img_rgb = img_rgb.transpose(_PILImage.FLIP_TOP_BOTTOM)
                        except Exception:
                            pass
                        img_rgb.save(str(out_path))
                        log.debug(f"Saved texture: {out_path.name}")
                    saved_files.append(out_path.name)
                except Exception as e:
                    failed.append(tex_name)
                    log.debug(f"Could not save texture '{tex_name}': {e}")
        if saved_files:
            log.info(f"Saved {len(saved_files)} texture(s) alongside export in {out_dir}")
        return TextureSidecarResult(
            requested_names=tuple(requested),
            saved_files=tuple(saved_files),
            missing_names=tuple(missing),
            failed_names=tuple(failed),
        )

    @staticmethod
    def _export_baked_lightmaps_to_dir(model: KotorModel, out_dir: Path) -> int:
        """Copy generated lightmap assets next to exported scene files."""
        saved = 0
        seen: set[str] = set()
        for node in model.mesh_nodes():
            path = str(getattr(node, "_gr_baked_lightmap_path", "") or "")
            if not path or path.lower() in seen:
                continue
            seen.add(path.lower())
            try:
                source = Path(path)
                if not source.is_file():
                    continue
                target = out_dir / source.name
                if source.resolve() != target.resolve():
                    target.write_bytes(source.read_bytes())
                saved += 1
            except Exception as exc:
                log.debug(f"Could not copy baked lightmap '{path}': {exc}")
        if saved:
            try:
                from src.core.lighting.lightmap_export_bridge import export_baked_lightmap_manifest

                export_baked_lightmap_manifest(model, out_dir)
            except Exception as exc:
                log.debug(f"Could not write baked lightmap export metadata: {exc}")
            log.info(f"Copied {saved} baked lightmap asset(s) alongside export in {out_dir}")
        return saved


def _renderable_mesh_nodes(model: KotorModel):
    """
    Return a list of mesh nodes from *model* that represent renderable geometry.
    This is the same filter used by OBJExporter and the viewport's
    _iter_visible_mesh_nodes().  Shared by OBJExporter and FBXExporter so both
    produce identical output meshes.

    Phase 15.3 FIX: Also include skin nodes (NodeFlags.SKIN = 0x0040).
    KotorModel.mesh_nodes() only returns nodes where is_mesh=True (flag 0x0020),
    silently skipping skin mesh nodes (flag 0x0040) that contain the creature's
    actual visible geometry (btBody_front, btBodyback, bthair, etc.).
    We call all_nodes() and check is_mesh OR is_skin to match the viewport's
    _iter_mesh_nodes() behaviour (Phase 16 fix).
    """
    nodes = []
    for n in model.all_nodes():
        if n.is_mesh or n.is_skin:
            if OBJExporter._is_renderable(n):
                nodes.append(n)
    return nodes


# Alias used by GLTFExporter (consistent naming across the exporter family)
_iter_visible_mesh_nodes = _renderable_mesh_nodes


# ──────────────────────────────────────────────────────────────────────
#  Rigging / Animation Sidecar Export
# ──────────────────────────────────────────────────────────────────────

def _export_rigging_data(model: KotorModel, out_dir: Path) -> int:
    """
    Export rigging and animation data as JSON sidecar files into
    ``out_dir/rigging/``.

    Files written
    -------------
    rigging/<model>.skeleton.json
        Complete bone hierarchy: name, parent, position (bind pose),
        orientation (quaternion), flags.

    rigging/<model>.weights.json
        Per-vertex skin weights for every skin mesh node.
        Format: {mesh_name: {vertex_index: [[bone_name, weight], ...]}}

    rigging/<model>.<anim_name>.anim.json  (one per animation)
        Animation keyframe data: length, transition_time, events,
        and per-node position + orientation curves.

    Returns the count of files successfully written (0 if model has
    no skeleton or animations).  Non-fatal: logs warnings on failure.

    The JSON sidecar format is intentionally human-readable and easy to
    consume by external scripts (e.g. Maya Python, Blender Python, Unity).
    """
    import json

    has_skin     = any(n.is_skin and n.bone_map
                       for n in model.all_nodes())
    has_anims    = bool(model.animations)
    # Write skeleton JSON only when the model has a meaningful rig.
    # A pure-mesh prop (root node + one non-skin mesh) has no rig data
    # worth exporting — the skeleton file would only contain the root and
    # the mesh node with no useful bone hierarchy.
    # Condition: at least one skin mesh OR at least one animation must exist.
    has_skeleton = (model.root_node is not None and (has_skin or has_anims))

    if not (has_skeleton or has_skin or has_anims):
        return 0

    rig_dir = out_dir / 'rigging'
    try:
        rig_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.warning(f"_export_rigging_data: cannot create {rig_dir}: {e}")
        return 0

    written = 0
    model_stem = model.name or 'model'

    # ── 1. Skeleton hierarchy ────────────────────────────────────────
    if has_skeleton:
        bones_list = []
        for n in model.all_nodes():
            bone_entry = {
                'name':        n.name,
                'parent':      n.parent.name if n.parent else None,
                'position':    list(n.position),
                'rotation':    list(n.rotation),  # (x,y,z,w) quaternion
                'flags':       int(n.flags),
                'is_skin':     bool(n.is_skin),
                'is_mesh':     bool(n.is_mesh),
            }
            bones_list.append(bone_entry)

        skel_data = {
            'model':       model_stem,
            'game':        str(getattr(model.game_version, 'name',
                                       model.game_version)),  # 'K1' or 'K2'
            'supermodel':  model.supermodel,
            'bone_count':  len(bones_list),
            'bones':       bones_list,
        }
        skel_path = rig_dir / f"{model_stem}.skeleton.json"
        try:
            skel_path.write_text(
                json.dumps(skel_data, indent=2, ensure_ascii=False),
                encoding='utf-8')
            written += 1
            log.debug(f"Skeleton JSON: {skel_path.name} ({len(bones_list)} bones)")
        except OSError as e:
            log.warning(f"_export_rigging_data: skeleton write failed: {e}")

    # ── 2. Skin weights ──────────────────────────────────────────────
    if has_skin:
        weights_data: dict = {}
        for n in model.all_nodes():
            if not (n.is_skin and n.bone_map and n.skin_data):
                continue
            mesh_weights: dict = {}
            for vi, sd in enumerate(n.skin_data):
                if not sd.influences:
                    continue
                inf_list = []
                for bw in sd.influences:
                    if bw.weight <= 0.0:
                        continue
                    # bone_index → bone name via bone_map
                    bi = bw.bone_index
                    bname = (n.bone_map[bi]
                             if 0 <= bi < len(n.bone_map) else f'bone_{bi}')
                    inf_list.append([bname, round(bw.weight, 6)])
                if inf_list:
                    mesh_weights[str(vi)] = inf_list
            if mesh_weights:
                weights_data[n.name] = mesh_weights

        if weights_data:
            wt_path = rig_dir / f"{model_stem}.weights.json"
            try:
                wt_path.write_text(
                    json.dumps(weights_data, indent=2, ensure_ascii=False),
                    encoding='utf-8')
                written += 1
                log.debug(f"Weights JSON: {wt_path.name} "
                          f"({len(weights_data)} mesh(es))")
            except OSError as e:
                log.warning(f"_export_rigging_data: weights write failed: {e}")

    # ── 3. Animations ────────────────────────────────────────────────
    for anim in model.animations:
        anim_nodes_data = []
        for an in anim.nodes:
            node_entry: dict = {
                'name':        an.name,
                'controllers': [],
            }
            for ctrl in an.controllers:
                ct = ctrl.get('type')
                times  = ctrl.get('times',  [])
                values = ctrl.get('values', [])
                node_entry['controllers'].append({
                    'type':   ct,
                    'times':  list(times),
                    'values': [list(v) if hasattr(v, '__iter__') else v
                               for v in values],
                })
            anim_nodes_data.append(node_entry)

        events_data = [{'time': ev.time, 'name': ev.name}
                       for ev in anim.events]

        anim_data = {
            'model':           model_stem,
            'name':            anim.name,
            'length':          anim.length,
            'transition_time': anim.transition_time,
            'anim_root':       anim.anim_root,
            'events':          events_data,
            'node_count':      len(anim_nodes_data),
            'nodes':           anim_nodes_data,
        }

        # Sanitise anim name for use as filename
        safe_name = ''.join(c if c.isalnum() or c in '_-' else '_'
                            for c in anim.name)[:64]
        anim_path = rig_dir / f"{model_stem}.{safe_name}.anim.json"
        try:
            anim_path.write_text(
                json.dumps(anim_data, indent=2, ensure_ascii=False),
                encoding='utf-8')
            written += 1
            log.debug(f"Anim JSON: {anim_path.name}  "
                      f"length={anim.length:.3f}s  "
                      f"{len(anim_nodes_data)} nodes")
        except OSError as e:
            log.warning(f"_export_rigging_data: anim write failed "
                        f"({anim.name}): {e}")

    return written


# ──────────────────────────────────────────────────────────────────────
#  FBX Exporter (via pyassimp or fbx SDK)
# ──────────────────────────────────────────────────────────────────────

class FBXExporter:
    """
    Export KotorModel to FBX.

    Strategy (priority order):
      1. If `fbx-python-sdk` (fbx module) is available: use it for proper binary FBX.
      2. If `pyassimp` is available: use assimp export pipeline.
      3. Fallback: write FBX ASCII 7.4 format manually (no external deps).
         This is readable by Blender, Maya, 3ds Max and most DCC tools.
         Includes full mesh geometry, UV, normals, skinning (bind pose +
         skin cluster deformers), and the bone hierarchy as null/joint nodes.
    """

    COMPATIBILITY_STANDARD = "standard"
    COMPATIBILITY_UNITY = "unity"
    COMPATIBILITY_UNREAL = "unreal"
    COMPATIBILITY_3DS_MAX = "3ds_max"

    @classmethod
    def normalize_compatibility_profile(cls, value: str | None) -> str:
        key = str(value or cls.COMPATIBILITY_STANDARD).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "": cls.COMPATIBILITY_STANDARD,
            "default": cls.COMPATIBILITY_STANDARD,
            "native": cls.COMPATIBILITY_STANDARD,
            "kotor": cls.COMPATIBILITY_STANDARD,
            "unity_engine": cls.COMPATIBILITY_UNITY,
            "unreal_engine": cls.COMPATIBILITY_UNREAL,
            "ue": cls.COMPATIBILITY_UNREAL,
            "ue5": cls.COMPATIBILITY_UNREAL,
            "max": cls.COMPATIBILITY_3DS_MAX,
            "3dsmax": cls.COMPATIBILITY_3DS_MAX,
            "autodesk_3ds_max": cls.COMPATIBILITY_3DS_MAX,
        }
        key = aliases.get(key, key)
        valid = {
            cls.COMPATIBILITY_STANDARD,
            cls.COMPATIBILITY_UNITY,
            cls.COMPATIBILITY_UNREAL,
            cls.COMPATIBILITY_3DS_MAX,
        }
        if key not in valid:
            raise ValueError(f"Unsupported FBX compatibility profile: {value}")
        return key

    def export(self, model: KotorModel, fbx_path: str, tex_cache=None,
               export_rigging: bool = True,
               base_skeleton_model: 'Optional[KotorModel]' = None,
               export_manifest: bool = True,
               compatibility_profile: str = "standard",
               force_ascii: bool = False) -> bool:
        """
        Export model to FBX.

        Parameters
        ----------
        model               : KotorModel to export.
        fbx_path            : Output .fbx file path.
        tex_cache           : Optional TextureCache.  When provided, textures are saved as
                              TGA files in the same directory as the FBX.
        export_rigging      : When True (default) a 'rigging/' subdirectory is created
                              next to the FBX file.  The FBX already embeds the skeleton
                              hierarchy + skin deformers + animations, so the rigging/
                              folder provides the same data as portable JSON files for
                              scripts that cannot read FBX directly.
        base_skeleton_model : Optional KotorModel containing the supermodel skeleton
                              (e.g. S_MALE02, S_FEMALE02).  When provided, synthetic
                              placeholder bones for cross-referenced supermodel bones
                              receive correct bind-pose transforms from this model
                              instead of identity matrices.  This is required for
                              accessory meshes (heads, bodies, hands) to deform
                              correctly in Unreal Engine.
        export_manifest     : When True (default), write a
                              ``<fbx_stem>.ghostrigger.json`` sidecar preserving
                              KOTOR node flags, hook roles, texture slots,
                              qbone/tbone skin data, animation summaries, and
                              Unreal import guidance that FBX cannot carry
                              reliably on its own.
        compatibility_profile : ``standard``, ``unity``, ``unreal``, or
                              ``3ds_max``. Engine/DCC profiles force the complete
                              ASCII writer and apply explicit unit and animation
                              interchange contracts.
        force_ascii         : Bypass optional SDK/Assimp writers and use the
                              built-in FBX 7.4 ASCII writer. Selection export
                              uses this to guarantee one merged geometry with
                              polygon material assignments and both UV channels.
        """
        compatibility_profile = self.normalize_compatibility_profile(compatibility_profile)
        exporter_backend = ""
        fbx_format = ""
        # Rigged or animated models must use the ASCII 7.4 writer: the SDK
        # path writes skeleton nodes translation-only (no rotation), builds no
        # skin clusters, and exports no animation takes, so a creature body
        # arrives in Max/Unity as scattered rigid parts (2026-07-15 user
        # report: republic soldier female).  The SDK path stays available for
        # static props, where a binary FBX is friendlier to Unity.
        has_skins = any(getattr(node, "is_skin", False) for node in model.all_nodes())
        has_animations = bool(getattr(model, "animations", None))
        if force_ascii or has_skins or has_animations or compatibility_profile != self.COMPATIBILITY_STANDARD:
            log.info(
                "FBX export: %s has %s; using the ASCII FBX writer (profile=%s; the SDK bridge "
                "does not export skin clusters, animation takes, or compatibility metadata yet).",
                model.name,
                (
                    "a forced ASCII selection export"
                    if force_ascii
                    else ("skinned meshes" if has_skins else ("animations" if has_animations else "a compatibility profile"))
                ),
                compatibility_profile,
            )
            ok = None
        else:
            # Try fbx module (Autodesk FBX Python SDK)
            try:
                import fbx as _fbx
                ok = self._export_fbx_sdk(model, fbx_path, _fbx)
                if ok:
                    exporter_backend = "autodesk_fbx_python_sdk"
                    fbx_format = "FBX SDK writer"
            except ImportError:
                ok = None

        if ok is None and compatibility_profile == self.COMPATIBILITY_STANDARD and not force_ascii:
            # Try pyassimp (treated as a hint only – fall through on any failure)
            try:
                import pyassimp  # noqa: F401
                _assimp_ok = self._export_assimp(model, fbx_path, pyassimp,
                                                 base_skeleton_model=base_skeleton_model)
                # _export_assimp now delegates to _export_fbx_ascii internally;
                # if it succeeds we are done, otherwise fall through.
                if _assimp_ok:
                    ok = True
                    exporter_backend = "pyassimp_ascii_delegate"
                    fbx_format = "FBX 7.4 ASCII"
                # If False, keep ok=None so the ASCII path runs below.
            except (ImportError, Exception):
                pass  # pyassimp not installed – go straight to ASCII
            except BaseException:
                pass  # pyassimp native lib missing (AssimpError is BaseException) – go to ASCII

        if ok is None:
            # Primary export path: FBX ASCII 7.4 (zero-dependency, full feature set)
            import traceback as _tb
            try:
                texture_paths = None
                if tex_cache is not None:
                    texture_paths = self._export_fbx_textures_to_dir(
                        model,
                        Path(fbx_path).parent / "textures",
                        tex_cache,
                        Path(fbx_path).parent,
                    )
                ascii_kwargs = {
                    "base_skeleton_model": base_skeleton_model,
                    "texture_paths": texture_paths,
                }
                # Preserve the long-standing monkeypatch/backend seam for the
                # standard writer while passing the new option only when needed.
                if compatibility_profile != self.COMPATIBILITY_STANDARD:
                    ascii_kwargs["compatibility_profile"] = compatibility_profile
                ok = self._export_fbx_ascii(model, fbx_path, **ascii_kwargs)
                if ok:
                    exporter_backend = "builtin_ascii"
                    fbx_format = "FBX 7.4 ASCII"
            except Exception as e:
                log.error(f"FBX ASCII export failed: {e}\n{_tb.format_exc()}")
                # Last resort: OBJ
                obj_path = str(Path(fbx_path).with_suffix('.obj'))
                log.warning(f"Falling back to OBJ export: {obj_path}")
                OBJExporter().export(model, obj_path, tex_cache=tex_cache,
                                     export_rigging=export_rigging)
                return False

        # Copy textures alongside the FBX for non-ASCII fallback backends.
        out_dir = Path(fbx_path).parent
        if ok and tex_cache is not None and exporter_backend != "builtin_ascii":
            OBJExporter._export_textures_to_dir(model, out_dir, tex_cache)
        if ok:
            OBJExporter._export_baked_lightmaps_to_dir(model, out_dir)

        # Export rigging + animations into a dedicated subfolder
        if ok and export_rigging:
            rig_count = _export_rigging_data(model, out_dir)
            if rig_count > 0:
                log.info(f"Rigging data exported: {rig_count} file(s) → "
                         f"{out_dir / 'rigging'}")
        if ok and export_manifest:
            try:
                from src.core.export.kotor_fbx_manifest import write_kotor_fbx_manifest

                exported_mesh_names = [node.name for node in _renderable_mesh_nodes(model)]
                manifest_path = write_kotor_fbx_manifest(
                    model,
                    fbx_path,
                    exported_mesh_names=exported_mesh_names,
                    exporter_backend=exporter_backend,
                    fbx_format=fbx_format,
                    target_engine=(
                        "unity" if compatibility_profile == self.COMPATIBILITY_UNITY else
                        "unreal" if compatibility_profile == self.COMPATIBILITY_UNREAL else
                        "3ds_max" if compatibility_profile == self.COMPATIBILITY_3DS_MAX else
                        "unreal"
                    ),
                    compatibility_profile=compatibility_profile,
                )
                log.info(f"GhostRigger FBX manifest exported: {manifest_path.name}")
            except Exception as exc:
                log.warning(f"Could not write GhostRigger FBX manifest: {exc}")
        return bool(ok)

    @staticmethod
    def _export_fbx_textures_to_dir(
        model: KotorModel,
        texture_dir: Path,
        tex_cache,
        fbx_dir: Path,
    ) -> Dict[str, str]:
        """Save FBX texture sidecars and return texture-name to relative path."""
        written: Dict[str, str] = {}
        if tex_cache is None:
            return written
        texture_dir.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        for node in model.mesh_nodes():
            material_nodes = list(getattr(node, "_gr_fbx_material_slots", None) or [node])
            for material_node in material_nodes:
                raw = (
                    getattr(material_node, "texture_clean", "")
                    or getattr(material_node, "texture", "")
                    or ""
                )
                tex_name = raw.strip()
                if not tex_name or tex_name.upper() in ("NULL", "BLACK"):
                    continue
                key = tex_name.lower()
                if key in seen:
                    continue
                seen.add(key)
                try:
                    img = tex_cache.get(tex_name)
                    if img is None:
                        continue
                    out_path = texture_dir / f"{tex_name}.png"
                    if not out_path.exists():
                        img_to_save = img.convert("RGBA") if getattr(img, "mode", "") != "RGBA" else img
                        transpose = getattr(img_to_save, "transpose", None)
                        if callable(transpose):
                            try:
                                from PIL import Image as _PILImage

                                img_to_save = transpose(_PILImage.FLIP_TOP_BOTTOM)
                            except Exception:
                                pass
                        img_to_save.save(str(out_path))
                    rel = os.path.relpath(out_path, fbx_dir).replace(os.sep, "/")
                    written[tex_name] = rel
                except Exception as exc:
                    log.debug("Could not save FBX texture '%s': %s", tex_name, exc)
        if written:
            log.info("Saved %d FBX texture sidecar(s) in %s", len(written), texture_dir)
        return written

    # ── FBX SDK (Autodesk Python SDK) ─────────────────────────────────

    def _export_fbx_sdk(self, model: KotorModel, fbx_path: str, fbx) -> bool:
        """Export using Autodesk FBX Python SDK if available."""
        try:
            mgr     = fbx.FbxManager.Create()
            scene   = fbx.FbxScene.Create(mgr, model.name)
            info    = fbx.FbxDocumentInfo.Create(mgr, "DocInfo")
            info.mTitle   = model.name
            info.mSubject = f"KotOR model exported by GhostRigger-K1-K2"
            scene.SetSceneInfo(info)

            # Set Z-up axis system (KotOR convention)
            axis_sys = fbx.FbxAxisSystem(
                fbx.FbxAxisSystem.eZAxis,
                fbx.FbxAxisSystem.eParityOdd,
                fbx.FbxAxisSystem.eRightHanded)
            scene.GetGlobalSettings().SetAxisSystem(axis_sys)

            root = scene.GetRootNode()
            node_map: Dict[str, 'fbx.FbxNode'] = {}

            # Create skeleton nodes
            for n in model.all_nodes():
                if n.is_dummy:
                    skel_attr = fbx.FbxSkeleton.Create(mgr, n.name)
                    skel_attr.SetSkeletonType(
                        fbx.FbxSkeleton.eRoot if n.parent is None
                        else fbx.FbxSkeleton.eLimbNode)
                    fbx_node = fbx.FbxNode.Create(mgr, n.name)
                    fbx_node.SetNodeAttribute(skel_attr)
                    fbx_node.LclTranslation.Set(fbx.FbxDouble3(*n.position))
                    node_map[n.name] = fbx_node

            # Build hierarchy for skeleton
            for n in model.all_nodes():
                if n.name in node_map:
                    fbx_node = node_map[n.name]
                    if n.parent and n.parent.name in node_map:
                        node_map[n.parent.name].AddChild(fbx_node)
                    else:
                        root.AddChild(fbx_node)

            # Create mesh nodes (only renderable geometry)
            for mesh_node in _renderable_mesh_nodes(model):
                self._add_fbx_mesh(mgr, scene, root, mesh_node, node_map)

            # Save
            exporter = fbx.FbxExporter.Create(mgr, "")
            ok = exporter.Initialize(fbx_path, -1, mgr.GetIOSettings())
            if ok:
                exporter.Export(scene)
            exporter.Destroy()
            mgr.Destroy()
            if ok:
                log.info(f"FBX SDK export: {fbx_path}")
                return True
        except Exception as e:
            log.error(f"FBX SDK export error: {e}")
        return False

    def _add_fbx_mesh(self, mgr, scene, root_fbx_node, mesh_node: ModelNode,
                      node_map: dict):
        """Add one KotOR mesh node to the FBX scene (FBX SDK path)."""
        try:
            import fbx
            fbx_mesh = fbx.FbxMesh.Create(mgr, mesh_node.name)

            # Vertices
            fbx_mesh.InitControlPoints(len(mesh_node.vertices))
            for i, (x,y,z) in enumerate(mesh_node.vertices):
                fbx_mesh.SetControlPointAt(fbx.FbxVector4(x,y,z,0), i)

            # Faces
            for face in mesh_node.faces:
                fbx_mesh.BeginPolygon(-1, -1, False)
                for vi in face: fbx_mesh.AddPolygon(vi)
                fbx_mesh.EndPolygon()

            # Normals
            if mesh_node.normals:
                nrm_layer = fbx.FbxLayerElementNormal.Create(fbx_mesh, "Normals")
                nrm_layer.SetMappingMode(fbx.FbxLayerElement.eByControlPoint)
                nrm_layer.SetReferenceMode(fbx.FbxLayerElement.eDirect)
                for nx,ny,nz in mesh_node.normals:
                    nrm_layer.GetDirectArray().Add(fbx.FbxVector4(nx,ny,nz,0))
                fbx_mesh.GetLayer(0).SetNormals(nrm_layer)

            # UVs
            if mesh_node.uvs:
                uv_layer = fbx.FbxLayerElementUV.Create(fbx_mesh, "UVMap")
                uv_layer.SetMappingMode(fbx.FbxLayerElement.eByControlPoint)
                uv_layer.SetReferenceMode(fbx.FbxLayerElement.eDirect)
                for u,v in mesh_node.uvs:
                    uv_layer.GetDirectArray().Add(fbx.FbxVector2(u, v))
                fbx_mesh.GetLayer(0).SetUVs(uv_layer)

            fbx_node = fbx.FbxNode.Create(mgr, mesh_node.name)
            fbx_node.SetNodeAttribute(fbx_mesh)
            root_fbx_node.AddChild(fbx_node)
        except Exception as e:
            log.debug(f"FBX SDK mesh add error ({mesh_node.name}): {e}")

    # ── pyassimp path ─────────────────────────────────────────────────

    def _export_assimp(self, model: KotorModel, fbx_path: str, pyassimp,
                       base_skeleton_model: 'Optional[KotorModel]' = None) -> bool:
        """Export via pyassimp (limited skeleton support)."""
        try:
            # pyassimp export is immature; fall through to ASCII
            log.info("pyassimp FBX export attempted but not reliable – using ASCII FBX")
            return self._export_fbx_ascii(model, fbx_path,
                                          base_skeleton_model=base_skeleton_model)
        except Exception as e:
            log.error(f"pyassimp export error: {e}")
            return False

    # ── FBX ASCII 7.4 (zero-dependency fallback) ──────────────────────

    def _export_fbx_ascii(self, model: KotorModel, fbx_path: str,
                          base_skeleton_model: 'Optional[KotorModel]' = None,
                          texture_paths: 'Optional[Dict[str, str]]' = None,
                          compatibility_profile: str = "standard") -> bool:
        """
        Write FBX ASCII 7.4 file.  No external dependencies required.
        Supported by Blender 2.79+, Maya 2016+, 3ds Max 2016+, Unreal Engine 5.

        Exports:
          - Geometry (vertices, normals, UVs per polygon vertex, polygon indices)
          - Material (diffuse/specular colour + texture reference)
          - Skeleton: ALL dummy/bone nodes exported as LimbNode (flags=0 or flags=HEADER)
          - Skin deformers with SubDeformer clusters per bone
          - Bind pose with correct column-major world matrices
          - Animations: position deltas * animscale + bind_pos, quaternion→Euler XYZ
          - Takes section for Blender / MotionBuilder compatibility

        KotorBlender-verified:
          - Position keyframes are DELTAS from rest position, scaled by model.anim_scale
          - Orientation keyframes are absolute quaternions [x,y,z,w]
          - Bone nodes have flags=0 (type_label='dummy'), root has flags=HEADER=0x01
          - Only nodes with type_label=='dummy' are skeleton joints (not renderable meshes)

        Parameters
        ----------
        base_skeleton_model : Optional KotorModel for the supermodel (e.g. S_MALE02).
            When provided, synthesised placeholder bones for cross-referenced
            supermodel joints receive correct bind-pose world matrices instead of
            identity matrices, fixing skin deformation in Unreal Engine.
        """
        compatibility_profile = self.normalize_compatibility_profile(compatibility_profile)
        texture_paths = texture_paths or {}
        unit_scale_factor = 100.0 if compatibility_profile in {
            self.COMPATIBILITY_UNITY,
            self.COMPATIBILITY_UNREAL,
            self.COMPATIBILITY_3DS_MAX,
        } else 1.0
        linear_animation_keys = compatibility_profile in {
            self.COMPATIBILITY_UNITY,
            self.COMPATIBILITY_UNREAL,
            self.COMPATIBILITY_3DS_MAX,
        }
        # Build a fast name→node lookup for the base skeleton (supermodel) so that
        # synthetic bones created for cross-referenced joints get correct transforms.
        _base_skel_node_by_name: Dict[str, 'ModelNode'] = {}
        if base_skeleton_model is not None:
            for _bsn in base_skeleton_model.all_nodes():
                _base_skel_node_by_name[_bsn.name.lower()] = _bsn
            log.debug(f"FBX export: base_skeleton_model '{base_skeleton_model.name}' "
                      f"loaded with {len(_base_skel_node_by_name)} nodes")
        lines: List[str] = []
        w = lines.append

        # Only export renderable mesh nodes (respects render flag + deform helpers)
        mesh_nodes_list = _renderable_mesh_nodes(model)
        material_nodes_by_mesh = {
            node.name: list(getattr(node, "_gr_fbx_material_slots", None) or [node])
            for node in mesh_nodes_list
        }
        renderable_mesh_names = {node.name for node in mesh_nodes_list}
        renderable_mesh_name_by_lower = {node.name.lower(): node.name for node in mesh_nodes_list}

        # ── Header ────────────────────────────────────────────────────
        # Keep ASCII FBX output reproducible so golden-file regression tests
        # identify real exporter drift instead of wall-clock noise.
        export_time = {
            "year": 2026,
            "month": 1,
            "day": 1,
            "hour": 0,
            "minute": 0,
            "second": 0,
        }
        w('; FBX 7.4.0 project file')
        w('; Created by GhostRigger-K1-K2')
        w(f'; Model: {model.name}')
        w(f'; Compatibility profile: {compatibility_profile}')
        w('')
        w('FBXHeaderExtension:  {')
        w('\tFBXHeaderVersion: 1003')
        w('\tFBXVersion: 7400')
        w(f'\tCreationTimeStamp:  {{')
        w(f'\t\tVersion: 1000')
        w(f'\t\tYear: {export_time["year"]}')
        w(f'\t\tMonth: {export_time["month"]}')
        w(f'\t\tDay: {export_time["day"]}')
        w(f'\t\tHour: {export_time["hour"]}')
        w(f'\t\tMinute: {export_time["minute"]}')
        w(f'\t\tSecond: {export_time["second"]}')
        w(f'\t\tMillisecond: 0')
        w('\t}')
        w(f'\tCreator: "GhostRigger-K1-K2 FBX Exporter"')
        w('}')
        w('')
        w('GlobalSettings:  {')
        w('\tVersion: 1000')
        w('\tProperties70:  {')
        # Z-up right-handed (KotOR coordinate system, same as UE5 import axis)
        # UE5 FbxMainImport.cpp ConvertScene() expects: Z-up, eParityOdd (neg Y front), RightHanded
        # When source axis matches UE5 import axis, no conversion matrix is applied → clean import.
        w('\t\tP: "UpAxis", "int", "Integer", "",2')
        w('\t\tP: "UpAxisSign", "int", "Integer", "",1')
        # Front axis: 1 = Y axis, FrontAxisSign=-1 means -Y forward (matches UE5 eParityOdd negation)
        w('\t\tP: "FrontAxis", "int", "Integer", "",1')
        w('\t\tP: "FrontAxisSign", "int", "Integer", "",-1')
        # Coord (right) axis: 0 = X axis, sign=1 → +X right
        w('\t\tP: "CoordAxis", "int", "Integer", "",0')
        w('\t\tP: "CoordAxisSign", "int", "Integer", "",1')
        # OriginalUpAxis: UE5 uses this to detect if an axis-flip was already baked in.
        # Value 2 = Z was the original up axis (same as UpAxis so no extra flip needed).
        w('\t\tP: "OriginalUpAxis", "int", "Integer", "",2')
        w('\t\tP: "OriginalUpAxisSign", "int", "Integer", "",1')
        # FBX system units are expressed in centimeters.  KOTOR character
        # coordinates are meter-scale (a humanoid is roughly 1.7 units tall),
        # so Unity/Unreal/Max compatibility profiles declare 100 cm per source unit.
        w(f'\t\tP: "UnitScaleFactor", "double", "Number", "",{unit_scale_factor:g}')
        w(f'\t\tP: "OriginalUnitScaleFactor", "double", "Number", "",{unit_scale_factor:g}')
        # Time / frame-rate settings (FBX 7.4 standard: TimeMode 6 = 30fps)
        # UE5 FbxMainImport.cpp reads GetTimeMode() to determine frame rate.
        # TimeMode 6 = eFBXTimeMode30 (30 fps) — standard for game animations.
        w('\t\tP: "TimeMode", "enum", "", "",6')
        w('\t\tP: "TimeProtocol", "enum", "", "",2')
        w('\t\tP: "SnapOnFrames", "bool", "", "",0')
        w('\t\tP: "ReferenceTimeIndex", "int", "Integer", "",-1')
        w('\t\tP: "TimelineInterpolateMode", "enum", "", "",1')
        w('\t\tP: "CustomFrameRate", "double", "Number", "",30')
        w('\t\tP: "CustomTimeMode", "enum", "", "",0')
        w('\t}')
        w('}')
        w('')

        # ── Documents section (mandatory for FBX 7.4 / UE5 / ufbx) ─────────
        # FBX 7.4 spec requires a Documents block containing at least one
        # Document (the "Scene").  Without this, UE5's FBX importer and
        # the ufbx reference parser may reject the file outright.
        # Cross-ref: Blender io_scene_fbx/export_fbx_bin.py fbx_documents_elements()
        w('Documents:  {')
        w('\tCount: 1')
        w('\tDocument: 1000000000, "", "Scene" {')
        w('\t\tProperties70:  {')
        w('\t\t\tP: "SourceObject", "object", "", ""')
        w('\t\t\tP: "ActiveAnimStackName", "KString", "", "", ""')
        w('\t\t}')
        w('\t\tRootNode: 0')
        w('\t}')
        w('}')
        w('')

        # ── ID helpers ────────────────────────────────────────────────
        _id_counter = [1000]
        def new_id():
            _id_counter[0] += 1
            return _id_counter[0]

        # Assign IDs
        node_ids:    Dict[str, int] = {}
        mesh_ids:    Dict[str, int] = {}
        mat_ids:     Dict[str, List[int]] = {}
        deform_ids:  Dict[str, int] = {}  # skin deformer per mesh
        cluster_ids: Dict[str, Dict[str,int]] = {}  # clusters[mesh_name][bone_name]
        bone_alias_ids: Dict[str, int] = {}
        bone_alias_attr_ids: Dict[str, int] = {}
        pose_id = new_id()

        for n in model.all_nodes():
            node_ids[n.name] = new_id()
        node_name_by_lower = {name.lower(): name for name in node_ids}

        def _node_name_for_ref(name: str) -> str | None:
            if not name:
                return None
            if name in node_ids:
                return name
            return node_name_by_lower.get(name.lower())

        def _renderable_mesh_name_for_ref(name: str) -> str | None:
            if not name:
                return None
            if name in renderable_mesh_names:
                return name
            return renderable_mesh_name_by_lower.get(name.lower())

        # RIGGING FIX: Some models (especially accessories / heads / equipment) rely
        # on bones provided by their supermodel skeleton (e.g. S_MALE02, S_FEMALE02).
        # Those bone nodes do NOT appear in this model's node tree, so they would be
        # silently skipped when building animation curves and skin clusters.
        #
        # Solution: scan all animation nodes AND all bone_map entries for names that
        # are referenced but missing from node_ids, and synthesise placeholder FBX
        # nodes for them.  This gives UE5 a complete skeleton reference even when
        # the supermodel bones are absent from the accessory model file.
        _extra_bone_nodes: Dict[str, ModelNode] = {}   # name \u2192 synthetic ModelNode
        _all_referenced_bones: set = set()
        # Collect from bone_map (skin cluster references)
        for mesh_n in mesh_nodes_list:
            if mesh_n.is_skin:
                for bname in (mesh_n.bone_map or []):
                    mesh_bone_name = _renderable_mesh_name_for_ref(bname)
                    node_bone_name = _node_name_for_ref(bname)
                    if bname and mesh_bone_name:
                        bone_alias_ids.setdefault(mesh_bone_name, new_id())
                    elif bname and node_bone_name is None:
                        _all_referenced_bones.add(bname)
        # Collect from animation node names
        for anim in model.animations:
            for an in anim.nodes:
                mesh_anim_name = _renderable_mesh_name_for_ref(an.name)
                if mesh_anim_name:
                    mesh_anim_node = next((n for n in mesh_nodes_list if n.name == mesh_anim_name), None)
                    if mesh_anim_node is not None and not mesh_anim_node.is_skin:
                        bone_alias_ids.setdefault(mesh_anim_name, new_id())
                elif an.name and _node_name_for_ref(an.name) is None:
                    _all_referenced_bones.add(an.name)
        # Preserve each referenced supermodel bone's ancestor chain.  Emitting
        # a child's base-skeleton LOCAL transform while parenting it directly
        # to the accessory root makes its hierarchy disagree with TransformLink
        # and BindPose world matrices.  Pulling only the required ancestors is
        # cheap and keeps standalone heads/equipment structurally coherent.
        if _base_skel_node_by_name:
            for referenced_name in tuple(_all_referenced_bones):
                base_node = _base_skel_node_by_name.get(str(referenced_name).lower())
                ancestor = getattr(base_node, "parent", None) if base_node is not None else None
                while ancestor is not None:
                    ancestor_name = str(getattr(ancestor, "name", "") or "")
                    if ancestor_name and _node_name_for_ref(ancestor_name) is None:
                        _all_referenced_bones.add(ancestor_name)
                    ancestor = getattr(ancestor, "parent", None)
        # Synthesise placeholder skeleton nodes for missing bones
        for bname in sorted(_all_referenced_bones):
            synth = ModelNode(name=bname, flags=0, position=(0.0, 0.0, 0.0),
                              rotation=(0.0, 0.0, 0.0, 1.0))
            # Find the root skeleton node to parent synthetic bones under
            root_sk = next((n for n in model.all_nodes()
                            if n.parent is None and n.type_label == 'dummy'), None)
            if root_sk:
                synth.parent = root_sk
            _extra_bone_nodes[bname] = synth
            node_ids[bname] = new_id()
            node_name_by_lower[bname.lower()] = bname
            log.debug(f"FBX export: synthesised missing bone node '{bname}' "
                      f"(supermodel reference)")

        for n in mesh_nodes_list:
            mesh_ids[n.name] = new_id()
            mat_ids[n.name] = [new_id() for _slot in material_nodes_by_mesh[n.name]]
            if n.is_skin:
                deform_ids[n.name] = new_id()
                cluster_ids[n.name] = {}
                for bname in n.bone_map:
                    if bname and bname not in cluster_ids[n.name]:
                        cluster_ids[n.name][bname] = new_id()

        def _bone_model_id(bname: str) -> int | None:
            """Return the FBX model ID to use for a skin-cluster bone link."""
            mesh_bone_name = _renderable_mesh_name_for_ref(bname)
            if mesh_bone_name in bone_alias_ids:
                return bone_alias_ids[mesh_bone_name]
            node_name = _node_name_for_ref(bname)
            if node_name is not None:
                return node_ids.get(node_name)
            return None

        def _fbx_uv_pair(node: ModelNode, u: float, v: float) -> tuple[float, float]:
            """Return the DCC-facing UV pair for this mesh node."""
            if not bool(getattr(node, "uv_v_flip", True)):
                return float(u), float(v)
            return float(u), 1.0 - float(v)

        # ── Definitions section (mandatory for FBX 7.4 / UE5 / ufbx) ─────
        # FBX 7.4 requires a Definitions block that declares the count of
        # each object type present in the file.  UE5 uses these counts to
        # pre-allocate internal arrays; ufbx uses them for schema validation.
        # Cross-ref: Blender io_scene_fbx/export_fbx_bin.py fbx_definitions_elements()
        #
        # We compute exact counts from the ID dicts that were just populated.
        _n_models    = len(node_ids)  # skeleton + mesh + synthetic
        _n_geometry  = len(mesh_ids)
        _n_material  = sum(len(ids) for ids in mat_ids.values())
        _n_nodeattr  = len(skel_attr_ids) if 'skel_attr_ids' in dir() else 0  # computed later
        _n_deformer  = len(deform_ids)
        _n_cluster   = sum(len(v) for v in cluster_ids.values())
        _n_texture   = 0  # counted below after tex objects are built
        _n_video     = 0
        _n_anim_stack = len(model.animations)
        _n_anim_layer = len(model.animations)
        # Exact anim curve/node counts require walking the animation tree;
        # we'll defer the Definitions block and write it via a placeholder.
        # Instead, write Definitions as a post-pass after all objects are built.
        # --- Mark position for Definitions insertion (placeholder approach) ---
        _definitions_insert_idx = len(lines)
        # (We insert the Definitions block here after counting all objects.)

        # ── Objects section ───────────────────────────────────────────
        w('Objects:  {')

        # Geometry objects
        for n in mesh_nodes_list:
            geo_id = mesh_ids[n.name]
            w(f'\tGeometry: {geo_id}, "Geometry::{n.name}", "Mesh" {{')

            # Vertices
            verts_flat = [c for v in n.vertices for c in v]
            w(f'\t\tVertices: *{len(verts_flat)} {{')
            w('\t\t\ta: ' + ','.join(f'{x:.6f}' for x in verts_flat))
            w('\t\t}')

            # Polygon vertex index (FBX uses negative last index per poly)
            poly_idx = []
            for face in n.faces:
                for i, vi in enumerate(face):
                    poly_idx.append(vi if i < len(face)-1 else -(vi+1))
            w(f'\t\tPolygonVertexIndex: *{len(poly_idx)} {{')
            w('\t\t\ta: ' + ','.join(str(i) for i in poly_idx))
            w('\t\t}')

            # Normals layer (ByPolygonVertex for per-face-corner normals)
            if n.normals:
                # Expand normals to per-polygon-vertex order
                nrm_poly = []
                for face in n.faces:
                    for vi in face:
                        if vi < len(n.normals):
                            nrm_poly.extend(n.normals[vi])
                        else:
                            nrm_poly.extend([0.0, 0.0, 1.0])
                w('\t\tLayerElementNormal: 0 {')
                w('\t\t\tVersion: 101')
                w('\t\t\tName: ""')
                w('\t\t\tMappingInformationType: "ByPolygonVertex"')
                w('\t\t\tReferenceInformationType: "Direct"')
                w(f'\t\t\tNormals: *{len(nrm_poly)} {{')
                w('\t\t\t\ta: ' + ','.join(f'{x:.6f}' for x in nrm_poly))
                w('\t\t\t}')
                w('\t\t}')

            # Tangents layer (ByPolygonVertex, Direct) — UE5 uses these for normal maps.
            # KotOR binary MDL stores per-vertex tangents in the MDX stream when bump maps
            # are used. We export them when available; UE5 recomputes them otherwise.
            _tangents_src = getattr(n, 'tangents', []) or []
            if _tangents_src and len(_tangents_src) > 0:
                tan_poly = []
                for face in n.faces:
                    for vi in face:
                        if vi < len(_tangents_src):
                            tan_poly.extend(_tangents_src[vi][:3])
                        else:
                            tan_poly.extend([1.0, 0.0, 0.0])
                # Binormal (bitangent) computed from normal × tangent per vertex
                bin_poly = []
                for face in n.faces:
                    for vi in face:
                        if vi < len(n.normals) and vi < len(_tangents_src):
                            nx, ny, nz = n.normals[vi][:3]
                            tx, ty, tz = _tangents_src[vi][:3]
                            # binormal = cross(normal, tangent)
                            bx = ny*tz - nz*ty
                            by = nz*tx - nx*tz
                            bz = nx*ty - ny*tx
                            bin_poly.extend([bx, by, bz])
                        else:
                            bin_poly.extend([0.0, 1.0, 0.0])
                w('\t\tLayerElementTangent: 0 {')
                w('\t\t\tVersion: 101')
                w('\t\t\tName: ""')
                w('\t\t\tMappingInformationType: "ByPolygonVertex"')
                w('\t\t\tReferenceInformationType: "Direct"')
                w(f'\t\t\tTangents: *{len(tan_poly)} {{')
                w('\t\t\t\ta: ' + ','.join(f'{x:.6f}' for x in tan_poly))
                w('\t\t\t}')
                w('\t\t}')
                w('\t\tLayerElementBinormal: 0 {')
                w('\t\t\tVersion: 101')
                w('\t\t\tName: ""')
                w('\t\t\tMappingInformationType: "ByPolygonVertex"')
                w('\t\t\tReferenceInformationType: "Direct"')
                w(f'\t\t\tBinormals: *{len(bin_poly)} {{')
                w('\t\t\t\ta: ' + ','.join(f'{x:.6f}' for x in bin_poly))
                w('\t\t\t}')
                w('\t\t}')

            # UV layer
            # KotORBlender insight: use ByPolygonVertex + IndexToDirect so that
            # UV seams (shared vertices with different UVs) are preserved correctly.
            # UE5 FBX importer requires this for proper texture projection.
            # FIX (face_uvs): KotOR ASCII MDL and some binary models have separate
            # tvert (texture vertex) indices in node.face_uvs that differ from the
            # vertex position indices.  Use them when present to avoid scrambled UVs.
            if n.uvs:
                # GhostRigger stores KotOR-oriented UVs internally.  Match the OBJ
                # and glTF exporters by converting V for DCC/engine importers.
                uv_flat = [c for u, v in n.uvs for c in _fbx_uv_pair(n, u, v)]
                _fuvs_fbx = getattr(n, 'face_uvs', []) or []
                _has_fuvs_fbx = bool(_fuvs_fbx) and len(_fuvs_fbx) == len(n.faces)
                _nuv_fbx = len(n.uvs)
                uv_idx = []
                for _fi_fbx, _face_fbx in enumerate(n.faces):
                    if _has_fuvs_fbx:
                        _fu = _fuvs_fbx[_fi_fbx]
                        for _k in range(3):
                            uv_idx.append(min(int(_fu[_k]), _nuv_fbx - 1) if _nuv_fbx > 0 else 0)
                    else:
                        for _vi in _face_fbx:
                            uv_idx.append(min(_vi, _nuv_fbx - 1) if _nuv_fbx > 0 else 0)
                w('\t\tLayerElementUV: 0 {')
                w('\t\t\tVersion: 101')
                w('\t\t\tName: "UVMap"')
                w('\t\t\tMappingInformationType: "ByPolygonVertex"')
                w('\t\t\tReferenceInformationType: "IndexToDirect"')
                w(f'\t\t\tUV: *{len(uv_flat)} {{')
                w('\t\t\t\ta: ' + ','.join(f'{x:.6f}' for x in uv_flat))
                w('\t\t\t}')
                w(f'\t\t\tUVIndex: *{len(uv_idx)} {{')
                w('\t\t\t\ta: ' + ','.join(str(i) for i in uv_idx))
                w('\t\t\t}')
                w('\t\t}')

            # Secondary UV layer (lightmap / UVMap_Lightmap)
            # Exported for area meshes and any node with a second UV set.
            _uvs_lm_n = getattr(n, 'uvs_lm', []) or getattr(n, 'uvs2', []) or []
            if _uvs_lm_n:
                _uv2_flat = [c for u, v in _uvs_lm_n for c in _fbx_uv_pair(n, u, v)]
                _nuv2 = len(_uvs_lm_n)
                _uv2_idx = []
                for _face2 in n.faces:
                    for _vi2 in _face2:
                        _uv2_idx.append(min(_vi2, _nuv2 - 1) if _nuv2 > 0 else 0)
                w('\t\tLayerElementUV: 1 {')
                w('\t\t\tVersion: 101')
                w('\t\t\tName: "UVMap_Lightmap"')
                w('\t\t\tMappingInformationType: "ByPolygonVertex"')
                w('\t\t\tReferenceInformationType: "IndexToDirect"')
                w(f'\t\t\tUV: *{len(_uv2_flat)} {{')
                w('\t\t\t\ta: ' + ','.join(f'{x:.6f}' for x in _uv2_flat))
                w('\t\t\t}')
                w(f'\t\t\tUVIndex: *{len(_uv2_idx)} {{')
                w('\t\t\t\ta: ' + ','.join(str(i) for i in _uv2_idx))
                w('\t\t\t}')
                w('\t\t}')

            # Material reference layer
            _material_count = len(material_nodes_by_mesh[n.name])
            _face_materials = [
                max(0, min(
                    int(n.face_mats[index]) if index < len(n.face_mats) else 0,
                    _material_count - 1,
                ))
                for index in range(len(n.faces))
            ]
            w('\t\tLayerElementMaterial: 0 {')
            w('\t\t\tVersion: 101')
            w('\t\t\tName: ""')
            w(
                '\t\t\tMappingInformationType: "ByPolygon"'
                if _material_count > 1
                else '\t\t\tMappingInformationType: "AllSame"'
            )
            w('\t\t\tReferenceInformationType: "IndexToDirect"')
            if _material_count > 1:
                w(f'\t\t\tMaterials: *{len(_face_materials)} {{')
                w('\t\t\t\ta: ' + ','.join(str(value) for value in _face_materials))
            else:
                w('\t\t\tMaterials: *1 {')
                w('\t\t\t\ta: 0')
            w('\t\t\t}')
            w('\t\t}')

            # Layer definition (layer 0 — primary geometry data)
            _uvs_lm_for_layer = getattr(n, 'uvs_lm', []) or getattr(n, 'uvs2', []) or []
            _tangents_for_layer = getattr(n, 'tangents', []) or []
            w('\t\tLayer: 0 {')
            w('\t\t\tVersion: 100')
            if n.normals:
                w('\t\t\tLayerElement:  {')
                w('\t\t\t\tType: "LayerElementNormal"')
                w('\t\t\t\tTypedIndex: 0')
                w('\t\t\t}')
            if _tangents_for_layer:
                w('\t\t\tLayerElement:  {')
                w('\t\t\t\tType: "LayerElementTangent"')
                w('\t\t\t\tTypedIndex: 0')
                w('\t\t\t}')
                w('\t\t\tLayerElement:  {')
                w('\t\t\t\tType: "LayerElementBinormal"')
                w('\t\t\t\tTypedIndex: 0')
                w('\t\t\t}')
            if n.uvs:
                w('\t\t\tLayerElement:  {')
                w('\t\t\t\tType: "LayerElementUV"')
                w('\t\t\t\tTypedIndex: 0')
                w('\t\t\t}')
            w('\t\t\tLayerElement:  {')
            w('\t\t\t\tType: "LayerElementMaterial"')
            w('\t\t\t\tTypedIndex: 0')
            w('\t\t\t}')
            w('\t\t}')
            # Layer 1: lightmap UV channel (when present)
            if _uvs_lm_for_layer:
                w('\t\tLayer: 1 {')
                w('\t\t\tVersion: 100')
                w('\t\t\tLayerElement:  {')
                w('\t\t\t\tType: "LayerElementUV"')
                w('\t\t\t\tTypedIndex: 1')
                w('\t\t\t}')
                w('\t\t}')

            w('\t}')  # end Geometry

        # Texture + Video objects (one per unique texture name)
        # FBX requires explicit Texture + Video objects to resolve material texture maps.
        # Without these, UE5 imports the geometry but leaves materials untextured.
        _tex_obj_ids: Dict[str, int] = {}   # tex_name → Texture object ID
        _tex_vid_ids: Dict[str, int] = {}   # tex_name → Video object ID
        _seen_tex_names: set = set()
        for n in mesh_nodes_list:
            for material_node in material_nodes_by_mesh[n.name]:
                tname_tex = (material_node.texture_clean or '').strip()
                if not (
                    tname_tex
                    and tname_tex.upper() not in ('NULL', 'BLACK', '')
                    and tname_tex not in _seen_tex_names
                ):
                    continue
                _seen_tex_names.add(tname_tex)
                # FBX wrap enum: 0 = Repeat, 1 = Clamp.  KotOR/TXI stores this
                # as clamp_s/clamp_t booleans; exporting it keeps Unity/Unreal
                # from guessing on large tiled menu and room UVs.
                wrap_u = 1 if bool(getattr(material_node, 'txi_clamp_s', False)) else 0
                wrap_v = 1 if bool(getattr(material_node, 'txi_clamp_t', False)) else 0
                tex_obj_id = new_id()
                vid_id     = new_id()
                _tex_obj_ids[tname_tex] = tex_obj_id
                _tex_vid_ids[tname_tex] = vid_id
                texture_file = texture_paths.get(tname_tex, f"{tname_tex}.tga")
                # Video (file reference)
                _n_video += 1
                w(f'\tVideo: {vid_id}, "Video::{tname_tex}", "Clip" {{')
                w(f'\t\tType: "Clip"')
                w(f'\t\tProperties70:  {{')
                w(f'\t\t\tP: "Path","KString","XRefUrl","","{texture_file}"')
                w(f'\t\t}}')
                w(f'\t\tUseMipMap: 0')
                w(f'\t\tFilename: "{texture_file}"')
                w(f'\t\tRelativeFilename: "{texture_file}"')
                w(f'\t}}')
                # Texture object
                _n_texture += 1
                w(f'\tTexture: {tex_obj_id}, "Texture::{tname_tex}", "" {{')
                w(f'\t\tType: "TextureVideoClip"')
                w(f'\t\tVersion: 202')
                w(f'\t\tTextureName: "{tname_tex}"')
                w(f'\t\tProperties70:  {{')
                w(f'\t\t\tP: "UVSet","KString","","","UVMap"')
                w(f'\t\t\tP: "UseMaterial","bool","","",1')
                w(f'\t\t\tP: "WrapModeU","enum","","",{wrap_u}')
                w(f'\t\t\tP: "WrapModeV","enum","","",{wrap_v}')
                w(f'\t\t}}')
                w(f'\t\tMedia: "{tname_tex}"')
                w(f'\t\tFileName: "{texture_file}"')
                w(f'\t\tRelativeFilename: "{texture_file}"')
                w(f'\t}}')

        # Material objects
        for n in mesh_nodes_list:
            for material_node, mid in zip(material_nodes_by_mesh[n.name], mat_ids[n.name]):
                tname = material_node.texture_clean or material_node.name
                w(f'\tMaterial: {mid}, "Material::{tname}", "" {{')
                w('\t\tVersion: 102')
                w(f'\t\tShadingModel: "Phong"')
                w('\t\tMultiLayer: 0')
                w('\t\tProperties70:  {')
                d = material_node.diffuse
                w(f'\t\t\tP: "DiffuseColor","ColorRGB","Color","",{d[0]:.4f},{d[1]:.4f},{d[2]:.4f}')
                s = material_node.specular
                w(f'\t\t\tP: "SpecularColor","ColorRGB","Color","",{s[0]:.4f},{s[1]:.4f},{s[2]:.4f}')
                w(f'\t\t\tP: "Shininess","double","Number","",{material_node.shininess:.2f}')
                w(f'\t\t\tP: "Opacity","double","Number","",{material_node.alpha:.4f}')
                w('\t\t}')
                w('\t}')  # end Material

        import math as _m

        def _quat_to_euler_deg(qx, qy, qz, qw):
            """
            Convert xyzw quaternion to Euler XYZ degrees for FBX.
            FBX/UE5 use Euler XYZ rotation order by default (applied Z, then Y, then X).
            This matches the standard intrinsic XYZ decomposition.
            KotorBlender stores orientation as [x, y, z, w] — same order we use here.
            """
            # Normalize
            mag = _m.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
            if mag > 1e-9:
                qx /= mag; qy /= mag; qz /= mag; qw /= mag
            # Intrinsic XYZ Euler decomposition
            sinr = 2*(qw*qx + qy*qz)
            cosr = 1 - 2*(qx*qx + qy*qy)
            ex = _m.degrees(_m.atan2(sinr, cosr))
            sinp = 2*(qw*qy - qz*qx)
            ey = _m.degrees(_m.asin(max(-1.0, min(1.0, sinp))))
            siny = 2*(qw*qz + qx*qy)
            cosy = 1 - 2*(qy*qy + qz*qz)
            ez = _m.degrees(_m.atan2(siny, cosy))
            return ex, ey, ez

        def _rotation_to_euler_deg(rotation):
            """Decompose a 3x3 rigid rotation using FBX's emitted XYZ branch."""
            sinp = max(-1.0, min(1.0, -float(rotation[2][0])))
            ey = _m.asin(sinp)
            if abs(abs(sinp) - 1.0) < 1e-8:
                # At gimbal lock keep Z at zero and retain the combined X term.
                ex = _m.atan2(-float(rotation[0][1]), float(rotation[1][1]))
                ez = 0.0
            else:
                ex = _m.atan2(float(rotation[2][1]), float(rotation[2][2]))
                ez = _m.atan2(float(rotation[1][0]), float(rotation[0][0]))
            return _m.degrees(ex), _m.degrees(ey), _m.degrees(ez)

        def _rotation_3x3(quat):
            qx, qy, qz, qw = (float(value) for value in quat)
            length = _m.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
            if length > 1e-9:
                qx, qy, qz, qw = qx/length, qy/length, qz/length, qw/length
            else:
                qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
            return (
                (1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)),
                (2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)),
                (2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)),
            )

        _world_bind_cache = {}

        def _world_bind_transform(node):
            """Compose the emitted FBX hierarchy using standard rigid math."""
            cache_key = id(node)
            cached = _world_bind_cache.get(cache_key)
            if cached is not None:
                return cached
            local_r = _rotation_3x3(getattr(node, "rotation", (0.0, 0.0, 0.0, 1.0)))
            local_t = tuple(float(value) for value in getattr(node, "position", (0.0, 0.0, 0.0)))
            parent = getattr(node, "parent", None)
            if parent is None:
                result = (local_t, local_r)
            else:
                parent_t, parent_r = _world_bind_transform(parent)
                world_r = tuple(
                    tuple(sum(parent_r[row][k] * local_r[k][col] for k in range(3)) for col in range(3))
                    for row in range(3)
                )
                world_t = tuple(
                    parent_t[row] + sum(parent_r[row][k] * local_t[k] for k in range(3))
                    for row in range(3)
                )
                result = (world_t, world_r)
            _world_bind_cache[cache_key] = result
            return result

        def _world_matrix_col_major(node) -> str:
            """
            Return 16 floats for FBX BindPose/TransformLink matrices.

            FBX ASCII matrix properties are serialized in row-major order,
            with translation in the last row for FBX's row-vector convention:
              [m00, m01, m02, m03,
               m10, m11, m12, m13,
               m20, m21, m22, m23,
               tx,  ty,  tz,  1]
            Unity's FBX importer reads Transform/TransformLink this way when
            constructing SkinnedMeshRenderer bindposes.

            Compose from the local transforms written to the FBX.  The generic
            model ``world_transform()`` helper uses a legacy point-rotation
            convention that diverges on some deep, near-180-degree joint
            chains, so it cannot define an interoperable FBX bind pose.
            """
            try:
                wp, world_r = _world_bind_transform(node)
                tx, ty, tz = wp
                mat = [world_r[0][0], world_r[0][1], world_r[0][2], 0.0,
                       world_r[1][0], world_r[1][1], world_r[1][2], 0.0,
                       world_r[2][0], world_r[2][1], world_r[2][2], 0.0,
                       tx,  ty,  tz,  1.0]
                return ','.join(f'{v:.6f}' for v in mat)
            except Exception:
                return '1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1'

        def _cluster_bind_matrix(node, bone_node) -> str:
            """Return Unity's raw mesh-to-bone cluster Transform matrix.

            Autodesk's SDK describes ``Transform`` and ``TransformLink`` as
            global bind matrices, but the serialized FBX cluster ``Transform``
            consumed by Blender and Unity is the mesh-to-bone bind matrix:

                inverse(bone_world_bind) * mesh_world_bind

            Unity stores that raw Transform as ``Mesh.bindposes``.  Writing the
            mesh world matrix directly therefore applies every bone transform
            twice and produces the classic exploded character on import.
            """
            try:
                mesh_pos, mesh_r = _world_bind_transform(node)
                bone_pos, bone_r = _world_bind_transform(bone_node)
                # inverse rigid rotation is transpose(bone_r).
                bind_r = tuple(
                    tuple(sum(bone_r[k][row] * mesh_r[k][col] for k in range(3)) for col in range(3))
                    for row in range(3)
                )
                delta = tuple(float(mesh_pos[index]) - float(bone_pos[index]) for index in range(3))
                bind_t = tuple(
                    sum(bone_r[k][row] * delta[k] for k in range(3))
                    for row in range(3)
                )
                values = [
                    bind_r[0][0], bind_r[1][0], bind_r[2][0], 0.0,
                    bind_r[0][1], bind_r[1][1], bind_r[2][1], 0.0,
                    bind_r[0][2], bind_r[1][2], bind_r[2][2], 0.0,
                    bind_t[0], bind_t[1], bind_t[2], 1.0,
                ]
                return ','.join(f'{value:.6f}' for value in values)
            except Exception:
                return _world_matrix_col_major(node)

        def _skeleton_display_size(node, *, fallback: float = 0.15) -> float:
            """Return a Blender-friendly display size for an FBX skeleton node."""
            try:
                candidates = []
                for child in getattr(node, "children", []) or []:
                    cx, cy, cz = getattr(child, "position", (0.0, 0.0, 0.0))
                    dist = _m.sqrt(float(cx) * float(cx) + float(cy) * float(cy) + float(cz) * float(cz))
                    if dist > 1e-5:
                        candidates.append(dist)
                if candidates:
                    size = min(candidates)
                else:
                    px, py, pz = getattr(node, "position", (0.0, 0.0, 0.0))
                    size = _m.sqrt(float(px) * float(px) + float(py) * float(py) + float(pz) * float(pz)) * 0.5
                if size <= 1e-5:
                    size = fallback
                return max(0.035, min(float(size), 0.85))
            except Exception:
                return fallback

        # ── Determine which nodes are skeleton joints ──────────────────────
        # v6.0 FIX: Include ALL non-renderable nodes as skeleton joints.
        # KotOR bone nodes include:
        #   1. type_label=='dummy' (flags=0 or flags=HEADER) — pure joint nodes
        #   2. Non-rendered trimesh nodes (bone proxies with null/empty texture)
        #      KotorBlender's is_char_bone() includes TRIMESH nodes with no render
        #      or null bitmap. These serve as skeleton bones in KotOR but were
        #      previously excluded from the FBX skeleton hierarchy.
        # Cross-ref: KotorBlender io_scene_kotor/utils.py is_char_bone() + is_char_dummy()
        #
        # v7.1 FIX-BONEFILTER (Finding 1.2 — KotorBlender utils.py cross-ref):
        # KotorBlender's is_char_bone() also checks classification == CHARACTER.
        # Non-CHARACTER models (effects, tiles) should not generate armature/skeleton.
        # We add this check: only include skeleton joints when the model's
        # classification is CHARACTER (4) or when it has skin mesh nodes.
        # This prevents tile/effect models from generating spurious bone hierarchies.
        _mesh_node_names = {n.name for n in mesh_nodes_list}
        # v7.2 FIX: Use None-check, not truthiness, so model_type=0 (EFFECT) is not
        # falsely promoted to CHARACTER (4).  0 is a valid classification.
        _mt_raw = getattr(model, 'model_type', None)
        _model_cls_int = int(_mt_raw) if _mt_raw is not None else 4
        _has_any_skin = any(n.is_skin for n in mesh_nodes_list)
        _is_character_model = (_model_cls_int == 4 or _has_any_skin)
        _skin_bone_names_lower = {
            str(bname).lower()
            for _skin_node in mesh_nodes_list
            if _skin_node.is_skin
            for bname in (_skin_node.bone_map or [])
            if bname
        }

        skeleton_nodes = []
        for n in model.all_nodes():
            if n.type_label == 'dummy':
                # v7.1: Only include dummy nodes in skeleton when model is CHARACTER
                # or has skin nodes. For tile/effect models, only the root node is
                # included (needed for FBX hierarchy).
                if _is_character_model or n.parent is None:
                    skeleton_nodes.append(n)
            elif (n.is_mesh and not n.is_skin and n.name not in _mesh_node_names
                  and not getattr(n, 'render', True)):
                # Non-rendered trimesh bone proxy — include in skeleton
                if _is_character_model:
                    skeleton_nodes.append(n)

        # v6.0 FIX: NodeAttribute IDs for skeleton nodes.
        # Unreal Engine requires NodeAttribute objects of type "Skeleton" attached
        # to each bone Model node. Without these, UE5 does not recognise the nodes
        # as skeleton joints and the Skeleton Editor shows no bones.
        # Cross-ref: ufbx.h ufbx_bone / FBX SDK FbxSkeleton::eLimbNode
        skel_attr_ids: Dict[str, int] = {}  # node name → NodeAttribute ID
        for n in skeleton_nodes:
            skel_attr_ids[n.name] = new_id()
        for bname in _extra_bone_nodes:
            skel_attr_ids[bname] = new_id()
        for bname in bone_alias_ids:
            bone_alias_attr_ids[bname] = new_id()

        # Skeleton/joint model nodes (root + all bone nodes)
        for n in skeleton_nodes:
            nid = node_ids[n.name]
            # Root node (flags=HEADER) → "Root"; child joint nodes → "LimbNode"
            is_root = (n.parent is None) or (n.flags == int(NodeFlags.HEADER))
            is_skin_influence = n.name.lower() in _skin_bone_names_lower
            fbx_node_type = 'LimbNode' if (not is_root or is_skin_influence) else 'Null'
            w(f'\tModel: {nid}, "Model::{n.name}", "{fbx_node_type}" {{')
            w('\t\tVersion: 232')
            w('\t\tProperties70:  {')
            px, py, pz = n.position
            w(f'\t\t\tP: "Lcl Translation","Lcl Translation","","A",{px:.6f},{py:.6f},{pz:.6f}')
            qx, qy, qz, qw = n.rotation
            ex, ey, ez = _quat_to_euler_deg(qx, qy, qz, qw)
            w(f'\t\t\tP: "Lcl Rotation","Lcl Rotation","","A",{ex:.4f},{ey:.4f},{ez:.4f}')
            w(f'\t\t\tP: "Lcl Scaling","Lcl Scaling","","A",1.000000,1.000000,1.000000')
            # Explicit rotation order (FBX default 0 = XYZ — same as KotOR/UE5 convention)
            w(f'\t\t\tP: "RotationOrder","enum","","",0')
            w('\t\t}')
            w('\t}')  # end Model (skeleton node)

        # v6.0 FIX: NodeAttribute objects for skeleton nodes.
        # FBX Skeleton NodeAttribute tells Unreal "this node is a bone".
        # Root skeleton gets Size=1, Type="Root"; children get Type="Limb".
        for n in skeleton_nodes:
            attr_id = skel_attr_ids[n.name]
            is_root = (n.parent is None) or (n.flags == int(NodeFlags.HEADER))
            limb_size = _skeleton_display_size(n, fallback=0.25 if is_root else 0.15)
            w(f'\tNodeAttribute: {attr_id}, "NodeAttribute::{n.name}", "LimbNode" {{')
            w(f'\t\tTypeFlags: "Skeleton"')
            w(f'\t\tProperties70:  {{')
            w(f'\t\t\tP: "Size", "double", "Number", "",{limb_size:.6f}')
            w(f'\t\t\tP: "LimbLength", "double", "Number", "",{limb_size:.6f}')
            w(f'\t\t}}')
            w(f'\t}}')

        # Synthetic (supermodel) bone nodes — emit as LimbNode stubs
        # These are referenced by skin clusters / animations but not in the model tree.
        # When base_skeleton_model is supplied we use the real bind-pose local transform
        # from that skeleton so that skin deformation is correct in Unreal Engine.
        for bname, synth_n in _extra_bone_nodes.items():
            nid = node_ids[bname]
            # Try to get real local transform from base skeleton
            _bskel_node = _base_skel_node_by_name.get(bname.lower())
            if _bskel_node is not None:
                _spx, _spy, _spz = _bskel_node.position
                _sqx, _sqy, _sqz, _sqw = _bskel_node.rotation
                _sex, _sey, _sez = _quat_to_euler_deg(_sqx, _sqy, _sqz, _sqw)
            else:
                _spx = _spy = _spz = 0.0
                _sex = _sey = _sez = 0.0
            w(f'\tModel: {nid}, "Model::{bname}", "LimbNode" {{')
            w('\t\tVersion: 232')
            w('\t\tProperties70:  {')
            w(f'\t\t\tP: "Lcl Translation","Lcl Translation","","A",{_spx:.6f},{_spy:.6f},{_spz:.6f}')
            w(f'\t\t\tP: "Lcl Rotation","Lcl Rotation","","A",{_sex:.4f},{_sey:.4f},{_sez:.4f}')
            w(f'\t\t\tP: "Lcl Scaling","Lcl Scaling","","A",1.000000,1.000000,1.000000')
            w(f'\t\t\tP: "RotationOrder","enum","","",0')
            w('\t\t}')
            w('\t}')  # end Model (synthetic bone stub)

        # v6.0 FIX: NodeAttribute for synthetic bones
        for bname in _extra_bone_nodes:
            attr_id = skel_attr_ids[bname]
            synth_size = 0.15
            w(f'\tNodeAttribute: {attr_id}, "NodeAttribute::{bname}", "LimbNode" {{')
            w(f'\t\tTypeFlags: "Skeleton"')
            w(f'\t\tProperties70:  {{')
            w(f'\t\t\tP: "Size", "double", "Number", "",{synth_size:.6f}')
            w(f'\t\t\tP: "LimbLength", "double", "Number", "",{synth_size:.6f}')
            w(f'\t\t}}')
            w(f'\t}}')

        # Renderable mesh nodes can also appear in KotOR bone maps. FBX DCCs
        # expect cluster links to target skeleton limbs, not mesh objects, so
        # create a separate limb alias while leaving the original node as Mesh.
        for bname, alias_id in bone_alias_ids.items():
            source_node = model.find_node(bname)
            if source_node is not None:
                px, py, pz = getattr(source_node, "position", (0.0, 0.0, 0.0))
                qx, qy, qz, qw = getattr(source_node, "rotation", (0.0, 0.0, 0.0, 1.0))
                ex, ey, ez = _quat_to_euler_deg(qx, qy, qz, qw)
            else:
                px = py = pz = ex = ey = ez = 0.0
            w(f'\tModel: {alias_id}, "Model::{bname}_bone", "LimbNode" {{')
            w('\t\tVersion: 232')
            w('\t\tProperties70:  {')
            w(f'\t\t\tP: "Lcl Translation","Lcl Translation","","A",{px:.6f},{py:.6f},{pz:.6f}')
            w(f'\t\t\tP: "Lcl Rotation","Lcl Rotation","","A",{ex:.4f},{ey:.4f},{ez:.4f}')
            w(f'\t\t\tP: "Lcl Scaling","Lcl Scaling","","A",1.000000,1.000000,1.000000')
            w(f'\t\t\tP: "RotationOrder","enum","","",0')
            w('\t\t}')
            w('\t}')

        for bname, attr_id in bone_alias_attr_ids.items():
            w(f'\tNodeAttribute: {attr_id}, "NodeAttribute::{bname}_bone", "LimbNode" {{')
            w(f'\t\tTypeFlags: "Skeleton"')
            w(f'\t\tProperties70:  {{')
            w(f'\t\t\tP: "Size", "double", "Number", "",0.150000')
            w(f'\t\t\tP: "LimbLength", "double", "Number", "",0.150000')
            w(f'\t\t}}')
            w(f'\t}}')

        # Mesh model nodes (non-dummy mesh nodes)
        for n in mesh_nodes_list:
            nid = node_ids[n.name]
            w(f'\tModel: {nid}, "Model::{n.name}", "Mesh" {{')
            w('\t\tVersion: 232')
            w('\t\tProperties70:  {')
            is_rigid_mesh_bone_alias = n.name in bone_alias_ids and not n.is_skin
            if n.is_skin:
                # Skinned mesh objects must not inherit animated bone transforms
                # through their scene parent while also being deformed by skin
                # clusters.  Emit them in bind-pose world space and connect them
                # under the model root; the Cluster Transform/TransformLink data
                # then supplies the intended bind relationship to the bones.
                (px, py, pz), mesh_world_rotation = _world_bind_transform(n)
                mex, mey, mez = _rotation_to_euler_deg(mesh_world_rotation)
            elif is_rigid_mesh_bone_alias:
                # KotOR can use a renderable rigid mesh as an animated bone
                # (for example the rancor jaw/eyes).  FBX importers need the
                # skeleton alias to own the transform; the visible mesh is
                # identity-local under that alias so it follows animation.
                px = py = pz = mex = mey = mez = 0.0
            else:
                px, py, pz = n.position
                mqx, mqy, mqz, mqw = getattr(n, "rotation", (0.0, 0.0, 0.0, 1.0))
                mex, mey, mez = _quat_to_euler_deg(mqx, mqy, mqz, mqw)
            w(f'\t\t\tP: "Lcl Translation","Lcl Translation","","A",{px:.6f},{py:.6f},{pz:.6f}')
            w(f'\t\t\tP: "Lcl Rotation","Lcl Rotation","","A",{mex:.4f},{mey:.4f},{mez:.4f}')
            w(f'\t\t\tP: "Lcl Scaling","Lcl Scaling","","A",1.000000,1.000000,1.000000')
            w('\t\t}')
            w('\t}')  # end Model (mesh node)

        # Skin deformers
        # v6.0 FIX: Weight normalization + zero-weight guard.
        # Mukundan (2022) §Vertex Blending: weights must sum to 1.0 per vertex.
        # Every vertex must have at least one bone influence.
        for n in mesh_nodes_list:
            if not n.is_skin: continue
            sid = deform_ids[n.name]
            w(f'\tDeformer: {sid}, "Deformer::{n.name}_Skin", "Skin" {{')
            w('\t\tVersion: 101')
            w('\t\tLink_DeformAcuracy: 50')
            w('\t}')

            # Sub-deformers (clusters per bone)
            # TransformLink = bone world-space bind matrix in FBX ASCII order
            # Transform = mesh node's world-space bind matrix (geometry_to_world).
            #   FBX spec: Transform brings vertices from mesh-local to bone-local.
            #   = inverse(mesh_world) for identity-bone case; for correct skinning
            #   UE5 uses: vertex_world = TransformLink^-1 * Transform * vertex_local
            #   For KotOR, skin vertices are in node-local space. Transform should
            #   be the mesh node's world matrix so UE5 can place them correctly.
            # v6.1 FIX: Use mesh node world matrix as Transform instead of identity.
            #   Cross-ref: ufbx.h ufbx_skin_cluster.mesh_node_to_bone
            #   Cross-ref: Mukundan (2022) Jk = Lk x Fk (bind pose formula)
            identity_m = '1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1'
            mesh_transform_m = _world_matrix_col_major(n)

            # v7.1 FIX-QBONETBONE (Finding 2.5 — reone mdlmdxreader.cpp cross-ref):
            # Build a qBone/tBone fallback matrix lookup from the skin node's stored
            # bind-pose arrays.  When world_transform() fails for a bone, use
            # qBone/tBone quaternion+translation to construct the bind matrix.
            # reone reads qBone quaternions AND tBone translations (lines 280-292),
            # then constructs per-bone matrices: M = translate(tBone) * rotate(qBone).
            _qbone_list = getattr(n, 'qbone_list', []) or []
            _tbone_list = getattr(n, 'tbone_list', []) or []

            def _qbone_matrix_col_major(bi: int) -> str:
                """Build FBX ASCII matrix from qBone/tBone arrays for bone index bi."""
                if bi < len(_qbone_list) and bi < len(_tbone_list):
                    qx, qy, qz, qw = _qbone_list[bi]
                    tx, ty, tz = _tbone_list[bi]
                    # Build rotation from quaternion
                    r00 = 1 - 2*(qy*qy + qz*qz); r01 = 2*(qx*qy - qz*qw); r02 = 2*(qx*qz + qy*qw)
                    r10 = 2*(qx*qy + qz*qw); r11 = 1 - 2*(qx*qx + qz*qz); r12 = 2*(qy*qz - qx*qw)
                    r20 = 2*(qx*qz - qy*qw); r21 = 2*(qy*qz + qx*qw); r22 = 1 - 2*(qx*qx + qy*qy)
                    mat = [r00, r01, r02, 0.0,
                           r10, r11, r12, 0.0,
                           r20, r21, r22, 0.0,
                           tx,  ty,  tz,  1.0]
                    return ','.join(f'{v:.6f}' for v in mat)
                return identity_m

            # Normalise skin_data: must be a list of VertexSkinData objects.
            raw_sd = getattr(n, 'skin_data', None)
            if not isinstance(raw_sd, (list, tuple)) or not raw_sd:
                raw_sd = []  # no per-vertex skin data – clusters will be empty
            else:
                if raw_sd and hasattr(raw_sd[0], 'bone_index'):
                    raw_sd = [VertexSkinData(influences=[bw]) for bw in raw_sd]
                elif raw_sd and not hasattr(raw_sd[0], 'influences'):
                    raw_sd = []

            # v7.0 FIX: Weight normalization pass + 4-influence limit.
            # Ensure all vertex weights sum to 1.0 and every vertex has at least
            # one bone influence (zero-weight guard).
            # Cross-ref: Mukundan (2022) — "Every vertex must have >= 1 bone influence"
            # v7.0 (Finding 1.4 — FBX2glTF FbxSkinningAccess.cpp cross-ref):
            #   FBX2glTF limits to MAX_WEIGHTS=4 per vertex (the FBX/UE5 standard).
            #   Sort influences by weight descending, keep top 4, re-normalize.
            #   This prevents UE5 import errors from vertices with >4 influences.
            _MAX_INFLUENCES = 4
            n_verts = len(n.vertices) if n.vertices else 0
            _norm_sd = list(raw_sd)  # copy for normalization
            for _vi in range(min(n_verts, len(_norm_sd))):
                sd = _norm_sd[_vi]
                if not sd.influences:
                    # Zero-weight guard: assign to bone 0 with weight 1.0
                    sd.influences = [BoneWeight(bone_index=0, weight=1.0)]
                    continue
                # Sort by weight descending, keep top MAX_INFLUENCES
                sd.influences.sort(key=lambda inf: inf.weight, reverse=True)
                if len(sd.influences) > _MAX_INFLUENCES:
                    sd.influences = sd.influences[:_MAX_INFLUENCES]
                # Normalize weights to sum to 1.0
                w_sum = sum(inf.weight for inf in sd.influences)
                if w_sum > 1e-6 and abs(w_sum - 1.0) > 1e-5:
                    for inf in sd.influences:
                        inf.weight /= w_sum
                elif w_sum < 1e-6:
                    sd.influences = [BoneWeight(bone_index=0, weight=1.0)]
            # Zero-weight guard for vertices beyond skin_data length
            while len(_norm_sd) < n_verts:
                _norm_sd.append(VertexSkinData(
                    influences=[BoneWeight(bone_index=0, weight=1.0)]))

            # v6.0 FIX: Emit ALL bone clusters, even empty ones.
            # Unreal Engine requires every bone referenced by the skeleton to have
            # a corresponding SubDeformer/Cluster in the Skin deformer, even if
            # that bone has zero direct vertex influence. Empty clusters keep the
            # skeleton hierarchy intact in UE5's Skeleton Editor.
            for bname, cid in cluster_ids.get(n.name, {}).items():
                bone_indices = [
                    bi for bi, candidate in enumerate(n.bone_map or [])
                    if candidate == bname
                ]
                if not bone_indices:
                    continue
                primary_bi = bone_indices[0]
                # Gather vertex indices + weights for this bone. Some KotOR skin
                # bone maps repeat the same bone name in multiple slots; FBX
                # needs one Cluster object per bone node, so merge those slots
                # into a single per-vertex weight list.
                weight_by_vertex = {}
                for vi, sd in enumerate(_norm_sd):
                    for inf in (sd.influences or []):
                        if inf.bone_index in bone_indices and inf.weight > 0:
                            weight_by_vertex[vi] = (
                                weight_by_vertex.get(vi, 0.0) + inf.weight
                            )
                vi_list = sorted(weight_by_vertex)
                wt_list = [weight_by_vertex[vi] for vi in vi_list]

                # TransformLink = bone world-space bind matrix (column-major for FBX)
                # Priority: (1) this model's own node, (2) base_skeleton_model node,
                # (3) identity fallback.
                bone_node = model.find_node(bname)
                if bone_node:
                    link_m = _world_matrix_col_major(bone_node)
                elif bname.lower() in _base_skel_node_by_name:
                    bone_node = _base_skel_node_by_name[bname.lower()]
                    link_m = _world_matrix_col_major(bone_node)
                elif primary_bi < len(_qbone_list) and primary_bi < len(_tbone_list):
                    # v7.1: qBone/tBone fallback (Finding 2.5)
                    link_m = _qbone_matrix_col_major(primary_bi)
                else:
                    link_m = identity_m
                cluster_transform_m = (
                    _cluster_bind_matrix(n, bone_node)
                    if bone_node is not None
                    and compatibility_profile == self.COMPATIBILITY_UNITY
                    else mesh_transform_m
                )

                # FBX cluster objects are Deformer records with subtype "Cluster".
                # Writing "SubDeformer:" produces an object our own text checks can
                # count, but Unity's FBX importer ignores for skin binding.
                w(f'\tDeformer: {cid}, "SubDeformer::{bname}", "Cluster" {{')
                w('\t\tVersion: 100')
                if vi_list:
                    w(f'\t\tIndexes: *{len(vi_list)} {{')
                    w('\t\t\ta: ' + ','.join(str(i) for i in vi_list))
                    w('\t\t}')
                    w(f'\t\tWeights: *{len(wt_list)} {{')
                    w('\t\t\ta: ' + ','.join(f'{x:.6f}' for x in wt_list))
                    w('\t\t}')
                # v6.1: Emit Transform (mesh world matrix) + TransformLink (bone world matrix).
                # Transform = mesh node world-space bind matrix. UE5 uses this to
                # convert vertices from geometry space to world space before applying
                # inverse bone transform. Using the mesh node's world matrix ensures
                # correct placement for skin nodes whose verts are in local space.
                w(f'\t\tTransform: *16 {{')
                w(f'\t\t\ta: {cluster_transform_m}')
                w('\t\t}')
                w(f'\t\tTransformLink: *16 {{')
                w(f'\t\t\ta: {link_m}')
                w('\t\t}')
                w('\t}')  # end Cluster deformer

        # Bind pose — include skeleton nodes + mesh nodes that have a world transform.
        # FBX spec: BindPose lists all nodes involved in skinning with their world matrices.
        # Using column-major matrix layout as required by FBX 7.4 / UE5 importer.
        # Include skeleton nodes, synthetic supermodel bone stubs, and mesh nodes.
        identity_bind = '1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1'
        pose_nodes = skeleton_nodes + mesh_nodes_list
        n_pose = len(pose_nodes) + len(_extra_bone_nodes) + len(bone_alias_ids)
        w(f'\tPose: {pose_id}, "Pose::BIND_POSES", "BindPose" {{')
        w(f'\t\tType: "BindPose"')
        w(f'\t\tVersion: 100')
        w(f'\t\tNbPoseNodes: {n_pose}')
        for n in pose_nodes:
            nid = node_ids[n.name]
            world_mat = _world_matrix_col_major(n)
            w(f'\t\tPoseNode:  {{')
            w(f'\t\t\tNode: {nid}')
            w(f'\t\t\tMatrix: *16 {{')
            w(f'\t\t\t\ta: {world_mat}')
            w(f'\t\t\t}}')
            w(f'\t\t}}')
        # Synthetic supermodel bones: use real world matrix from base skeleton when
        # available; otherwise fall back to identity (position/orientation unknown).
        for bname, _synth in _extra_bone_nodes.items():
            nid = node_ids[bname]
            _bsn2 = _base_skel_node_by_name.get(bname.lower())
            synth_mat = _world_matrix_col_major(_bsn2) if _bsn2 else identity_bind
            w(f'\t\tPoseNode:  {{')
            w(f'\t\t\tNode: {nid}')
            w(f'\t\t\tMatrix: *16 {{')
            w(f'\t\t\t\ta: {synth_mat}')
            w(f'\t\t\t}}')
            w(f'\t\t}}')
        w('\t}')  # end Pose

        for bname, alias_id in bone_alias_ids.items():
            alias_node = model.find_node(bname)
            alias_mat = _world_matrix_col_major(alias_node) if alias_node is not None else identity_bind
            insert_at = len(lines) - 1
            alias_pose = [
                f'\t\tPoseNode:  {{',
                f'\t\t\tNode: {alias_id}',
                f'\t\t\tMatrix: *16 {{',
                f'\t\t\t\ta: {alias_mat}',
                f'\t\t\t}}',
                f'\t\t}}',
            ]
            for offset, pose_line in enumerate(alias_pose):
                lines.insert(insert_at + offset, pose_line)

        # ── Animation objects (AnimStack / AnimLayer / AnimCurve) ──────────────
        # Build animation data before closing Connections; we need to know all IDs
        # to emit proper OO/OP connection records.
        #
        # FBX time unit: 1 second = 46186158000 ticks (standard FBX 7.4 rate)
        FBX_TICKS_PER_SEC = 46186158000

        # anim_connections accumulates Connection lines emitted AFTER the
        # per-animation Objects are written; they are flushed into Connections{}.
        anim_connections: List[str] = []

        # Build model-level node name→node lookup ONCE (outside animation loop)
        _base_node_map = {n.name.lower(): n for n in model.all_nodes()}

        if model.animations:
            CTRL_POSITION    = 8
            CTRL_ORIENTATION = 20

            anim_stack_layer: List[tuple] = []  # (anim, stack_id, layer_id)

            for anim in model.animations:
                # Include animations even when anim.nodes is empty – the AnimStack
                # and Takes entries must still be written so UE5 sees all clip names.
                stack_id = new_id()
                layer_id = new_id()
                anim_stack_layer.append((anim, stack_id, layer_id))

                # Unreal's importer uses the "|<name>" stack convention to
                # distinguish multiple takes. Unity exposes that separator as
                # part of the AnimationClip name, so keep Unity/Max/standard
                # stack names clean and reserve the prefix for Unreal only.
                anim_length_ticks = int(anim.length * FBX_TICKS_PER_SEC)
                stack_take_name = (
                    f"|{anim.name}"
                    if compatibility_profile == self.COMPATIBILITY_UNREAL
                    else str(anim.name)
                )
                fbx_stack_name = f"AnimStack::{stack_take_name}"
                w(f'\tAnimationStack: {stack_id}, "{fbx_stack_name}", "" {{')
                w(f'\t\tProperties70:  {{')
                w(f'\t\t\tP: "LocalStart", "KTime", "Time", "",0')
                w(f'\t\t\tP: "LocalStop", "KTime", "Time", "",{anim_length_ticks}')
                w(f'\t\t\tP: "ReferenceStart", "KTime", "Time", "",0')
                w(f'\t\t\tP: "ReferenceStop", "KTime", "Time", "",{anim_length_ticks}')
                w(f'\t\t}}')
                w(f'\t}}')  # end AnimationStack

                # AnimationLayer
                w(f'\tAnimationLayer: {layer_id}, "AnimLayer::{anim.name}_Layer", "" {{')
                w(f'\t}}')  # end AnimationLayer

                if not anim.nodes:
                    continue  # no keyframe data – stack is valid but empty

                anim_conn = anim_connections  # shortcut

                for anim_node in anim.nodes:
                    base = _base_node_map.get(anim_node.name.lower())
                    nid = _bone_model_id(anim_node.name)
                    if nid is None:
                        continue

                    # Collect controllers
                    # Controllers may be stored as a list of dicts {'type','times','values'}
                    # (MDL parser format) OR as a dict keyed by type int (legacy format).
                    pos_times = pos_vals = None
                    rot_times = rot_vals = None
                    _ctrl_src = anim_node.controllers
                    if isinstance(_ctrl_src, dict):
                        # Legacy dict format: {8: {'times':..,'values':..}, 20: ...}
                        _ctrl_iter = [{'type': k, **v} for k, v in _ctrl_src.items()]
                    else:
                        _ctrl_iter = list(_ctrl_src or [])
                    for ctrl in _ctrl_iter:
                        ct = ctrl['type']
                        if ct == CTRL_POSITION:
                            pos_times, pos_vals = ctrl['times'], ctrl['values']
                        elif ct == CTRL_ORIENTATION:
                            rot_times, rot_vals = ctrl['times'], ctrl['values']

                    def _write_anim_curve(cv_id, default_val, ktimes, kvals,
                                          cn_id, ax_letter):
                        """Write a single AnimationCurve object (Blender/UE5-compatible).

                        FBX 7.4 spec (verified against Blender io_scene_fbx source):
                          - KeyAttrFlags:    *1  {a: 24776}   one flag for entire curve
                          - KeyAttrDataFloat: *4 {a:0,0,0,0}  tangent data (cubic)
                          - KeyAttrRefCount:  *1 {a: N}       N = number of keys
                        """
                        nt = len(ktimes)
                        ticks = [int(t * FBX_TICKS_PER_SEC) for t in ktimes]
                        w(f'\tAnimationCurve: {cv_id}, "AnimCurve::", "" {{')
                        w(f'\t\tDefault: {default_val:.6f}')
                        w(f'\t\tKeyVer: 4008')
                        w(f'\t\tKeyTime: *{nt} {{')
                        w('\t\t\ta: ' + ','.join(str(t) for t in ticks))
                        w(f'\t\t}}')
                        w(f'\t\tKeyValueFloat: *{nt} {{')
                        w('\t\t\ta: ' + ','.join(f'{v:.6f}' for v in kvals))
                        w(f'\t\t}}')
                        # Compatibility exports use linear keys.  This matches
                        # Odyssey's sampled controller semantics and avoids cubic
                        # tangent overshoot in Unity, Unreal, and 3ds Max on facial/limb bones.
                        w(f'\t\tKeyAttrFlags: *1 {{')
                        w(f'\t\t\ta: {4 if linear_animation_keys else 24776}')
                        w(f'\t\t}}')
                        # KeyAttrDataFloat: tangent data (4 floats — required for cubic mode)
                        w(f'\t\tKeyAttrDataFloat: *4 {{')
                        w('\t\t\ta: 0,0,0,0')
                        w(f'\t\t}}')
                        w(f'\t\tKeyAttrRefCount: *1 {{')
                        w(f'\t\t\ta: {nt}')
                        w(f'\t\t}}')
                        w(f'\t}}')  # end AnimationCurve
                        # Queue connections (emitted later in Connections{})
                        anim_conn.append(f'\tC: "OP",{cv_id},{cn_id},"d|{ax_letter}"')

                    def _write_grouped_curves(prop_channel, kvals_xyz,
                                              ktimes, def_vals, prop_name):
                        """Write one AnimationCurveNode (T or R) with 3 AnimationCurves.

                        Blender's FBX importer (io_scene_fbx/import_fbx.py) expects:
                          - ONE AnimationCurveNode per transform channel (T, R, S)
                          - The CurveNode name must be the channel letter only: 'T' or 'R'
                          - Properties70 must contain d|X, d|Y, d|Z defaults
                          - Three separate AnimationCurve objects, each linked via OP connection
                        """
                        cn_id = new_id()
                        cx_id = new_id()
                        cy_id = new_id()
                        cz_id = new_id()
                        dx, dy, dz = def_vals
                        # One CurveNode for all 3 axes
                        w(f'\tAnimationCurveNode: {cn_id}, "AnimCurveNode::{prop_channel}", "" {{')
                        w(f'\t\tProperties70:  {{')
                        w(f'\t\t\tP: "d|X", "Number", "", "A",{dx:.6f}')
                        w(f'\t\t\tP: "d|Y", "Number", "", "A",{dy:.6f}')
                        w(f'\t\t\tP: "d|Z", "Number", "", "A",{dz:.6f}')
                        w(f'\t\t}}')
                        w(f'\t}}')
                        # Three AnimationCurve objects
                        _write_anim_curve(cx_id, dx, ktimes, kvals_xyz[0], cn_id, 'X')
                        _write_anim_curve(cy_id, dy, ktimes, kvals_xyz[1], cn_id, 'Y')
                        _write_anim_curve(cz_id, dz, ktimes, kvals_xyz[2], cn_id, 'Z')
                        # CurveNode → AnimLayer connection
                        anim_conn.append(f'\tC: "OO",{cn_id},{layer_id}')
                        # CurveNode → bone model-node property
                        anim_conn.append(f'\tC: "OP",{cn_id},{nid},"{prop_name}"')

                    # Translation curves:
                    # KotOR position keyframes are DELTAS from the node's rest/bind position,
                    # scaled by model.anim_scale (verified from KotorBlender animnode.py:
                    #   bl_location = restloc + animscale * mdl_position_delta)
                    # Absolute FBX position = bind_pos + anim_scale * keyframe_delta
                    anim_scale = getattr(model, 'anim_scale', 1.0) or 1.0
                    if pos_times and pos_vals:
                        bind_pos = list(base.position) if base else [0.0, 0.0, 0.0]
                        kvals_t = []
                        for axis_i in range(3):
                            kvals_t.append([
                                (v[axis_i] if len(v) > axis_i else 0.0) * anim_scale
                                + bind_pos[axis_i]
                                for v in pos_vals
                            ])
                        _write_grouped_curves('T', kvals_t, pos_times,
                                              (bind_pos[0], bind_pos[1], bind_pos[2]),
                                              'Lcl Translation')

                    # Rotation curves:
                    # KotOR orientation keyframes are local absolute quaternions.
                    # KotorBlender converts them to pose-bone deltas because Blender
                    # stores a separate edit-bone rest orientation.  FBX Model
                    # "Lcl Rotation" is the actual local transform property, so
                    # animation curves must write absolute local Euler values.
                    if rot_times and rot_vals:
                        # Pre-compute Euler angles from absolute local quaternions.
                        # Unity, Unreal, and Max consume FBX Euler curves, so keep each axis
                        # on the nearest equivalent branch instead of allowing a
                        # +179 -> -179 discontinuity to spin/deform the skeleton.
                        euler_list = []
                        for qv in rot_vals:
                            if len(qv) >= 4:
                                ex, ey, ez = _quat_to_euler_deg(qv[0], qv[1], qv[2], qv[3])
                                euler_list.append((ex, ey, ez))
                            else:
                                euler_list.append((0.0, 0.0, 0.0))
                        if linear_animation_keys and len(euler_list) > 1:
                            continuous = [euler_list[0]]
                            for current in euler_list[1:]:
                                previous = continuous[-1]
                                adjusted = []
                                for value, prior in zip(current, previous):
                                    value = float(value)
                                    while value - prior > 180.0:
                                        value -= 360.0
                                    while value - prior < -180.0:
                                        value += 360.0
                                    adjusted.append(value)
                                continuous.append(tuple(adjusted))
                            euler_list = continuous
                        kvals_r = [
                            [e[axis_i] for e in euler_list]
                            for axis_i in range(3)
                        ]
                        def_r = (kvals_r[0][0] if kvals_r[0] else 0.0,
                                 kvals_r[1][0] if kvals_r[1] else 0.0,
                                 kvals_r[2][0] if kvals_r[2] else 0.0)
                        _write_grouped_curves('R', kvals_r, rot_times, def_r,
                                              'Lcl Rotation')

            # AnimStack → AnimLayer connections (added to anim_connections)
            for anim, stack_id, layer_id in anim_stack_layer:
                anim_connections.append(f'\tC: "OO",{layer_id},{stack_id}')

        w('}')  # end Objects

        # ── Definitions section (deferred — insert at saved position) ──────────
        # Now that all objects are emitted, we know exact counts for each type.
        # Build the Definitions block and splice it into `lines` at the saved
        # position (_definitions_insert_idx) so it appears BEFORE Objects.
        # This two-pass approach avoids forward-counting complexity while keeping
        # the final FBX output structurally correct.
        #
        # FBX 7.4 requires Definitions to appear between GlobalSettings/Documents
        # and Objects.  UE5's FBX importer reads ObjectType counts to pre-allocate
        # internal arrays.  ufbx uses them for validation.
        # Cross-ref: Blender io_scene_fbx/export_fbx_bin.py fbx_definitions_elements()
        _n_nodeattr  = len(skel_attr_ids) + len(bone_alias_attr_ids)
        _n_models    = len(node_ids) + len(bone_alias_ids)
        _n_geometry  = len(mesh_ids)
        _n_material  = len(mat_ids)
        _n_deformer  = len(deform_ids) + sum(len(v) for v in cluster_ids.values())
        _n_pose      = 1 if (skeleton_nodes or mesh_nodes_list) else 0
        # Count anim objects from what was actually emitted
        _n_anim_stack = 0
        _n_anim_layer = 0
        _n_anim_curvenode = 0
        _n_anim_curve = 0
        for _ln in lines:
            if '\tAnimationStack:' in _ln:
                _n_anim_stack += 1
            elif '\tAnimationLayer:' in _ln:
                _n_anim_layer += 1
            elif '\tAnimationCurveNode:' in _ln:
                _n_anim_curvenode += 1
            elif '\tAnimationCurve:' in _ln:
                _n_anim_curve += 1

        _defs: List[str] = []
        _da = _defs.append
        # Count distinct ObjectTypes present (GlobalSettings always counts as 1)
        _type_count = 1  # GlobalSettings
        _obj_types = [
            ('Model',              _n_models),
            ('Geometry',           _n_geometry),
            ('Material',           _n_material),
            ('Texture',            _n_texture),
            ('Video',              _n_video),
            ('NodeAttribute',      _n_nodeattr),
            ('Deformer',           _n_deformer),
            ('Pose',               _n_pose),
            ('AnimationStack',     _n_anim_stack),
            ('AnimationLayer',     _n_anim_layer),
            ('AnimationCurveNode', _n_anim_curvenode),
            ('AnimationCurve',     _n_anim_curve),
        ]
        for _otype, _ocount in _obj_types:
            if _ocount > 0:
                _type_count += 1

        _da('Definitions:  {')
        _da(f'\tVersion: 100')
        _da(f'\tCount: {_type_count}')
        # GlobalSettings template (always present)
        _da('\tObjectType: "GlobalSettings" {')
        _da('\t\tCount: 1')
        _da('\t}')
        for _otype, _ocount in _obj_types:
            if _ocount > 0:
                _da(f'\tObjectType: "{_otype}" {{')
                _da(f'\t\tCount: {_ocount}')
                _da('\t}')
        _da('}')
        _da('')
        # Splice into lines at the saved position
        for _di, _dl in enumerate(_defs):
            lines.insert(_definitions_insert_idx + _di, _dl)

        # ── Takes section (legacy Blender / MotionBuilder / UE4 compatibility) ─────
        # FBX 7.4 readers such as Blender's FBX importer, MotionBuilder, and UE4 use
        # the Takes block to enumerate animation clips.  UE5 uses AnimStack directly
        # but emitting Takes keeps compatibility with the full DCC pipeline.
        # All animations (even those without keyframe nodes) must be listed here so
        # that UE5 sees the complete animation library from the KotOR base skeleton.
        w('')
        w('Takes:  {')
        if model.animations:
            first_anim = model.animations[0]
            w(f'\tCurrent: "{first_anim.name}"')
            for anim in model.animations:
                anim_length_ticks_t = int(anim.length * FBX_TICKS_PER_SEC)
                w(f'\tTake: "{anim.name}" {{')
                w(f'\t\tFileName: "{anim.name}.tak"')
                w(f'\t\tLocalTime: 0,{anim_length_ticks_t}')
                w(f'\t\tReferenceTime: 0,{anim_length_ticks_t}')
                w(f'\t}}')
        else:
            w('\tCurrent: ""')
        w('}')

        # ── Connections section ────────────────────────────────────────
        w('')
        w('Connections:  {')

        # Node → parent hierarchy (model nodes)
        _root_nid_for_skin_meshes = 0
        _root_for_skin_meshes = next((n for n in model.all_nodes() if n.parent is None), None)
        if _root_for_skin_meshes and _root_for_skin_meshes.name in node_ids:
            _root_nid_for_skin_meshes = node_ids[_root_for_skin_meshes.name]
        _mesh_nodes_by_name = {n.name: n for n in mesh_nodes_list}
        for n in model.all_nodes():
            nid = node_ids[n.name]
            if n.name in _mesh_nodes_by_name and _mesh_nodes_by_name[n.name].is_skin:
                w(f'\tC: "OO",{nid},{_root_nid_for_skin_meshes}')
            elif n.name in bone_alias_ids and n.name in _mesh_nodes_by_name:
                w(f'\tC: "OO",{nid},{bone_alias_ids[n.name]}')
            elif n.parent and n.parent.name in node_ids:
                pid = node_ids[n.parent.name]
                w(f'\tC: "OO",{nid},{pid}')
            else:
                w(f'\tC: "OO",{nid},0')

        # v6.0 FIX: NodeAttribute → Model connections for skeleton nodes.
        # Each skeleton NodeAttribute must be connected to its Model node via OO.
        for n in skeleton_nodes:
            if n.name in skel_attr_ids:
                w(f'\tC: "OO",{skel_attr_ids[n.name]},{node_ids[n.name]}')

        # Synthetic supermodel bones retain their real base-skeleton hierarchy.
        # The highest missing ancestor is attached beneath this model's root.
        _root_nid = 0
        _root_n = next((n for n in model.all_nodes() if n.parent is None), None)
        if _root_n and _root_n.name in node_ids:
            _root_nid = node_ids[_root_n.name]
        for bname in _extra_bone_nodes:
            nid = node_ids[bname]
            parent_id = _root_nid
            base_node = _base_skel_node_by_name.get(bname.lower())
            base_parent_name = str(
                getattr(getattr(base_node, "parent", None), "name", "") or ""
            )
            exported_parent_name = _node_name_for_ref(base_parent_name)
            if exported_parent_name and exported_parent_name != bname:
                parent_id = node_ids[exported_parent_name]
            w(f'\tC: "OO",{nid},{parent_id}')
            # v6.0 FIX: NodeAttribute → synthetic bone connections
            if bname in skel_attr_ids:
                w(f'\tC: "OO",{skel_attr_ids[bname]},{nid}')

        for bname, alias_id in bone_alias_ids.items():
            source_node = model.find_node(bname)
            parent_id = _root_nid
            parent = getattr(source_node, "parent", None) if source_node is not None else None
            parent_name = getattr(parent, "name", "")
            parent_alias_name = _renderable_mesh_name_for_ref(parent_name)
            parent_node_name = _node_name_for_ref(parent_name)
            if parent_alias_name in bone_alias_ids:
                parent_id = bone_alias_ids[parent_alias_name]
            elif parent_node_name is not None:
                parent_id = node_ids[parent_node_name]
            w(f'\tC: "OO",{alias_id},{parent_id}')
            if bname in bone_alias_attr_ids:
                w(f'\tC: "OO",{bone_alias_attr_ids[bname]},{alias_id}')

        # Geometry → mesh model node
        for n in mesh_nodes_list:
            w(f'\tC: "OO",{mesh_ids[n.name]},{node_ids[n.name]}')

        # Material → mesh model node
        for n in mesh_nodes_list:
            for material_id in mat_ids[n.name]:
                w(f'\tC: "OO",{material_id},{node_ids[n.name]}')

        # Skin deformer → geometry
        for n in mesh_nodes_list:
            if not n.is_skin: continue
            w(f'\tC: "OO",{deform_ids[n.name]},{mesh_ids[n.name]}')
            # Clusters → skin deformer
            for bname, cid in cluster_ids.get(n.name, {}).items():
                w(f'\tC: "OO",{cid},{deform_ids[n.name]}')
                # Cluster → bone joint node
                bone_id = _bone_model_id(bname)
                if bone_id is not None:
                    w(f'\tC: "OO",{bone_id},{cid}')

        # Texture → material connections
        # Links each Texture object to its Material via the "DiffuseColor" property slot.
        for n in mesh_nodes_list:
            for material_node, material_id in zip(material_nodes_by_mesh[n.name], mat_ids[n.name]):
                tname = (material_node.texture_clean or '').strip()
                if tname and tname in _tex_obj_ids:
                    tex_oid = _tex_obj_ids[tname]
                    w(f'\tC: "OP",{tex_oid},{material_id},"DiffuseColor"')
        # Video → Texture connections
        for tname, tex_oid in _tex_obj_ids.items():
            vid_id = _tex_vid_ids.get(tname)
            if vid_id:
                w(f'\tC: "OO",{vid_id},{tex_oid}')

        # Animation curve connections (built above in Objects section loop)
        for conn_line in anim_connections:
            w(conn_line)

        w('}')  # end Connections

        # Write to file
        content = '\n'.join(lines)
        with open(fbx_path, 'w', encoding='utf-8') as f:
            f.write(content)

        log.info(f"FBX ASCII export: {fbx_path} "
                 f"({len(mesh_nodes_list)} meshes, "
                 f"{len(model.all_nodes())} nodes, "
                 f"{len(model.animations)} anims, "
                 f"{len(content)} bytes)")
        return True



# ──────────────────────────────────────────────────────────────────────
#  TGA ↔ TPC Texture Converter
# ──────────────────────────────────────────────────────────────────────

TPC_GREY = 1; TPC_RGB = 2; TPC_RGBA = 4; TPC_DXT1 = 12; TPC_DXT5 = 14
TPC_HDR  = 128

def tga_to_tpc(tga_path: str, tpc_path: str, txi_str: str = "", mipmaps: bool = True) -> bool:
    try:
        with open(tga_path,'rb') as f: raw = f.read()
        id_len,cm_type,img_type = struct.unpack_from('<BBB',raw,0)
        w,h = struct.unpack_from('<HH',raw,12)
        bpp = raw[16]
        desc= raw[17]
        off = 18 + id_len

        # Handle RLE-compressed TGA (types 9,10,11)
        is_rle = img_type in (9,10,11)
        is_bw  = img_type in (3,11)

        if is_rle:
            px = bpp//8; total = w*h
            pxdata = bytearray()
            pos = off
            while len(pxdata) < total*px and pos < len(raw):
                rep = raw[pos]; pos+=1
                if rep & 0x80:  # run
                    cnt = (rep&0x7F)+1
                    p   = raw[pos:pos+px]; pos+=px
                    pxdata.extend(p*cnt)
                else:           # raw
                    cnt = rep+1
                    pxdata.extend(raw[pos:pos+cnt*px]); pos+=cnt*px
            pixel_data = bytes(pxdata)
        else:
            pixel_data = raw[off:]

        # Flip vertically if origin is top-left (bit 5 of descriptor = 0 means bottom-left)
        row_sz = w * (bpp//8)
        rows   = [pixel_data[i*row_sz:(i+1)*row_sz] for i in range(h)]
        if not (desc & 0x20):  # bottom-left origin
            rows = list(reversed(rows))
        pixel_data = b''.join(rows)

        # Convert BGR(A) → RGB(A) and determine TPC encoding
        px = bpp//8
        converted = bytearray()
        if bpp == 32:
            for i in range(0, len(pixel_data), 4):
                b,g,r,a = pixel_data[i],pixel_data[i+1],pixel_data[i+2],pixel_data[i+3]
                converted.extend([r,g,b,a])
            enc = TPC_RGBA
        elif bpp == 24:
            for i in range(0, len(pixel_data), 3):
                b,g,r = pixel_data[i],pixel_data[i+1],pixel_data[i+2]
                converted.extend([r,g,b])
            enc = TPC_RGB
        elif bpp == 8:
            converted.extend(pixel_data); enc = TPC_GREY
        else:
            return False

        # Build mip chain
        mips = [bytes(converted)]
        if mipmaps:
            mips = _gen_mips(bytes(converted), w, h, bpp//8)

        # TPC header (128 bytes)
        total_sz = sum(len(m) for m in mips)
        hdr = bytearray(TPC_HDR)
        struct.pack_into('<I',hdr,0,total_sz)
        struct.pack_into('<H',hdr,8,w)
        struct.pack_into('<H',hdr,10,h)
        hdr[12] = enc
        hdr[13] = len(mips)

        with open(tpc_path,'wb') as f:
            f.write(hdr)
            for m in mips: f.write(m)
            if txi_str: f.write(txi_str.encode('ascii'))
        return True
    except Exception as e:
        log.error(f"TGA→TPC failed: {e}")
        return False


def tpc_to_tga(tpc_path: str, tga_path: str) -> bool:
    """
    Convert KotOR TPC texture to TGA.
    Supports uncompressed (Grey/RGB/RGBA) and DXT1/DXT5 block-compressed formats.
    KotOR TPC uses data_size field to identify compression:
      data_sz == w*h     → Grey (enc=1)
      data_sz == w*h*3   → RGB (enc=2 when uncompressed)
      data_sz == w*h*4   → RGBA (enc=4 when uncompressed)
      data_sz == (w//4)*(h//4)*8   → DXT1
      data_sz == (w//4)*(h//4)*16  → DXT5
    """
    try:
        with open(tpc_path,'rb') as f: data = f.read()
        if len(data) < TPC_HDR:
            return False
        data_sz = struct.unpack_from('<I',data,0)[0]
        w = struct.unpack_from('<H',data,8)[0]
        h = struct.unpack_from('<H',data,10)[0]
        enc = data[12]

        if w == 0 or h == 0:
            return False

        off = TPC_HDR
        pixel_data = data[off:]

        # Identify format by comparing data_sz to expected sizes
        dxt1_sz = max(8, (max(1,w//4)) * (max(1,h//4)) * 8)
        dxt5_sz = max(16, (max(1,w//4)) * (max(1,h//4)) * 16)

        if data_sz == w * h:
            # Greyscale
            raw = pixel_data[:w*h]
            img_type = 3; tga_bpp = 8
            converted = bytearray(raw)
        elif data_sz == w * h * 3:
            # RGB
            raw = pixel_data[:w*h*3]
            converted = bytearray()
            for i in range(0, len(raw), 3):
                r,g,b = raw[i],raw[i+1],raw[i+2]
                converted.extend([b,g,r])
            img_type = 2; tga_bpp = 24
        elif data_sz == w * h * 4:
            # RGBA
            raw = pixel_data[:w*h*4]
            converted = bytearray()
            for i in range(0, len(raw), 4):
                r,g,b,a = raw[i],raw[i+1],raw[i+2],raw[i+3]
                converted.extend([b,g,r,a])
            img_type = 2; tga_bpp = 32
        elif data_sz == dxt1_sz:
            # DXT1 block-compressed
            rgba = _decompress_dxt1(pixel_data, w, h)
            converted = bytearray()
            for i in range(0, len(rgba), 4):
                r,g,b,a = rgba[i],rgba[i+1],rgba[i+2],rgba[i+3]
                converted.extend([b,g,r,a])
            img_type = 2; tga_bpp = 32
        elif data_sz == dxt5_sz:
            # DXT5 block-compressed
            rgba = _decompress_dxt5(pixel_data, w, h)
            converted = bytearray()
            for i in range(0, len(rgba), 4):
                r,g,b,a = rgba[i],rgba[i+1],rgba[i+2],rgba[i+3]
                converted.extend([b,g,r,a])
            img_type = 2; tga_bpp = 32
        else:
            # Fallback: try uncompressed based on enc field
            bpp = {1:1, 2:3, 4:4}.get(enc, 3)
            expected = w * h * bpp
            if len(pixel_data) < expected:
                log.error(f"TPC→TGA: insufficient data for {w}x{h} enc={enc}")
                return False
            raw = pixel_data[:expected]
            if bpp == 4:
                converted = bytearray()
                for i in range(0,len(raw),4):
                    r,g,b,a=raw[i],raw[i+1],raw[i+2],raw[i+3]
                    converted.extend([b,g,r,a])
                img_type=2; tga_bpp=32
            elif bpp == 3:
                converted = bytearray()
                for i in range(0,len(raw),3):
                    r,g,b=raw[i],raw[i+1],raw[i+2]
                    converted.extend([b,g,r])
                img_type=2; tga_bpp=24
            else:
                converted = bytearray(raw); img_type=3; tga_bpp=8

        hdr = struct.pack('<BBBHHHHHHHBB',0,0,img_type,0,0,0,0,0,w,h,tga_bpp,0x20)
        with open(tga_path,'wb') as f:
            f.write(hdr); f.write(bytes(converted))
        return True
    except Exception as e:
        log.error(f"TPC→TGA failed: {e}"); return False


def _decompress_dxt1(data: bytes, w: int, h: int) -> bytes:
    """Software DXT1 decompressor. Returns RGBA bytes."""
    result = bytearray(w * h * 4)
    blocks_x = max(1, (w + 3) // 4)
    blocks_y = max(1, (h + 3) // 4)
    pos = 0
    for by in range(blocks_y):
        for bx in range(blocks_x):
            if pos + 8 > len(data): break
            c0r = struct.unpack_from('<H', data, pos)[0]
            c1r = struct.unpack_from('<H', data, pos+2)[0]
            lk  = struct.unpack_from('<I', data, pos+4)[0]
            pos += 8
            def e565(c):
                return (((c>>11)&0x1F)*255//31, ((c>>5)&0x3F)*255//63, (c&0x1F)*255//31)
            c0 = e565(c0r); c1 = e565(c1r)
            if c0r > c1r:
                cols = [c0,c1,
                        ((2*c0[0]+c1[0])//3,(2*c0[1]+c1[1])//3,(2*c0[2]+c1[2])//3),
                        ((c0[0]+2*c1[0])//3,(c0[1]+2*c1[1])//3,(c0[2]+2*c1[2])//3)]
            else:
                cols = [c0,c1,
                        ((c0[0]+c1[0])//2,(c0[1]+c1[1])//2,(c0[2]+c1[2])//2),
                        (0,0,0)]
            for py2 in range(4):
                for px2 in range(4):
                    idx=(lk>>(2*(py2*4+px2)))&3; col=cols[idx]
                    gx=bx*4+px2; gy=by*4+py2
                    if gx<w and gy<h:
                        o=(gy*w+gx)*4
                        result[o]=col[0]; result[o+1]=col[1]; result[o+2]=col[2]; result[o+3]=255
    return bytes(result)


def _decompress_dxt5(data: bytes, w: int, h: int) -> bytes:
    """Software DXT5 decompressor. Returns RGBA bytes."""
    result = bytearray(w * h * 4)
    blocks_x = max(1, (w + 3) // 4)
    blocks_y = max(1, (h + 3) // 4)
    pos = 0
    for by in range(blocks_y):
        for bx in range(blocks_x):
            if pos + 16 > len(data): break
            a0=data[pos]; a1=data[pos+1]
            abits=struct.unpack_from('<Q',data,pos+1)[0]>>8
            pos+=8
            c0r=struct.unpack_from('<H',data,pos)[0]; c1r=struct.unpack_from('<H',data,pos+2)[0]
            lk=struct.unpack_from('<I',data,pos+4)[0]; pos+=8
            def e565(c):
                return (((c>>11)&0x1F)*255//31,((c>>5)&0x3F)*255//63,(c&0x1F)*255//31)
            c0=e565(c0r); c1=e565(c1r)
            cols=[c0,c1,((2*c0[0]+c1[0])//3,(2*c0[1]+c1[1])//3,(2*c0[2]+c1[2])//3),
                        ((c0[0]+2*c1[0])//3,(c0[1]+2*c1[1])//3,(c0[2]+2*c1[2])//3)]
            if a0>a1: als=[a0,a1,(6*a0+a1)//7,(5*a0+2*a1)//7,(4*a0+3*a1)//7,(3*a0+4*a1)//7,(2*a0+5*a1)//7,(a0+6*a1)//7]
            else: als=[a0,a1,(4*a0+a1)//5,(3*a0+2*a1)//5,(2*a0+3*a1)//5,(a0+4*a1)//5,0,255]
            for py2 in range(4):
                for px2 in range(4):
                    ci=(lk>>(2*(py2*4+px2)))&3; ai=(abits>>(3*(py2*4+px2)))&7
                    col=cols[ci]; alpha=als[ai]
                    gx=bx*4+px2; gy=by*4+py2
                    if gx<w and gy<h:
                        o=(gy*w+gx)*4
                        result[o]=col[0]; result[o+1]=col[1]; result[o+2]=col[2]; result[o+3]=alpha
    return bytes(result)


def _gen_mips(data: bytes, w: int, h: int, bpp: int) -> List[bytes]:
    mips = [data]
    cur = bytearray(data)
    cw, ch = w, h
    while cw > 1 or ch > 1:
        nw, nh = max(1,cw>>1), max(1,ch>>1)
        nxt = bytearray(nw*nh*bpp)
        for y in range(nh):
            for x in range(nw):
                for c in range(bpp):
                    s = []
                    for dy in range(2):
                        for dx in range(2):
                            sx=min(x*2+dx,cw-1); sy=min(y*2+dy,ch-1)
                            s.append(cur[(sy*cw+sx)*bpp+c])
                    nxt[(y*nw+x)*bpp+c] = sum(s)//4
        mips.append(bytes(nxt)); cur=nxt; cw,ch=nw,nh
    return mips


# ──────────────────────────────────────────────────────────────────────
#  GLTF 2.0 / GLB Importer
# ──────────────────────────────────────────────────────────────────────

class GLTFImporter:
    """
    Import GLTF 2.0 / GLB files into KotorModel.

    Uses pygltflib if available (preferred), otherwise falls back to
    trimesh which also supports GLTF/GLB.

    KotOR UV convention: V is stored bottom-up. GLTF stores V top-down
    (same as DirectX), so we flip V on import: v_kotor = 1.0 - v_gltf.
    """

    def import_file(self, path: str,
                    model_name: str = "",
                    game_version: GameVersion = GameVersion.K1,
                    supermodel: str = "NULL",
                    classification: str = "character") -> Optional[KotorModel]:
        if not model_name:
            model_name = Path(path).stem[:32]
        try:
            return self._load_pygltflib(path, model_name, game_version, supermodel, classification)
        except ImportError:
            pass
        try:
            return self._load_trimesh(path, model_name, game_version, supermodel, classification)
        except ImportError:
            pass
        log.error("GLTF import: install 'pygltflib' or 'trimesh'")
        return None

    def _load_pygltflib(self, path, model_name, gv, sm, cl) -> KotorModel:
        import pygltflib
        import numpy as np
        import struct as st

        gltf = pygltflib.GLTF2().load(path)
        model = KotorModel(name=model_name, supermodel=sm, game_version=gv, classification=cl)
        root  = ModelNode(name=model_name, flags=int(NodeFlags.HEADER))
        model.root_node = root

        def _get_accessor_data(acc_idx):
            """Decode GLTF accessor data into numpy array."""
            if acc_idx is None:
                return None
            acc = gltf.accessors[acc_idx]
            bv  = gltf.bufferViews[acc.bufferView]
            buf = gltf.buffers[bv.buffer]
            # Get raw bytes
            raw = gltf.get_data_from_buffer_uri(buf.uri) if buf.uri else bytes(gltf.binary_blob())
            offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
            count  = acc.count
            type_map = {
                'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4,
                'MAT2': 4, 'MAT3': 9, 'MAT4': 16
            }
            comp_map = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
                        5123: np.uint16, 5125: np.uint32, 5126: np.float32}
            n_comp = type_map.get(acc.type, 1)
            dtype  = comp_map.get(acc.componentType, np.float32)
            stride = bv.byteStride or (np.dtype(dtype).itemsize * n_comp)
            result = []
            for i in range(count):
                row = []
                for j in range(n_comp):
                    off2 = offset + i * stride + j * np.dtype(dtype).itemsize
                    val  = st.unpack_from('<' + ('f' if dtype == np.float32 else
                                                 ('I' if dtype == np.uint32 else
                                                  ('H' if dtype == np.uint16 else 'B'))),
                                          raw, off2)[0]
                    row.append(val)
                result.append(row[0] if n_comp == 1 else tuple(row))
            return result

        for gnode in (gltf.nodes or []):
            nm = (gnode.name or "node")[:32]
            tx, ty, tz = 0.0, 0.0, 0.0
            if gnode.translation:
                tx, ty, tz = float(gnode.translation[0]), float(gnode.translation[1]), float(gnode.translation[2])
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
            if gnode.rotation:
                qx, qy, qz, qw = (float(gnode.rotation[0]), float(gnode.rotation[1]),
                                   float(gnode.rotation[2]), float(gnode.rotation[3]))

            flags = int(NodeFlags.HEADER)
            mnode = ModelNode(name=nm, flags=flags, position=(tx, ty, tz),
                              rotation=(qx, qy, qz, qw), parent=root)
            root.children.append(mnode)

            if gnode.mesh is not None:
                gmesh = gltf.meshes[gnode.mesh]
                for prim in (gmesh.primitives or []):
                    pnm = (gmesh.name or nm)[:32]
                    mesh_node = ModelNode(name=pnm, flags=int(NodeFlags.HEADER | NodeFlags.MESH),
                                         parent=mnode)
                    attrs = prim.attributes

                    pos_data = _get_accessor_data(getattr(attrs, 'POSITION', None))
                    if pos_data:
                        mesh_node.vertices = [(float(v[0]), float(v[1]), float(v[2]))
                                              for v in pos_data]
                    norm_data = _get_accessor_data(getattr(attrs, 'NORMAL', None))
                    if norm_data:
                        mesh_node.normals = [(float(n[0]), float(n[1]), float(n[2]))
                                             for n in norm_data]
                    uv_data = _get_accessor_data(getattr(attrs, 'TEXCOORD_0', None))
                    if uv_data:
                        # GLTF UV: V is top-down → flip to KotOR bottom-up
                        mesh_node.uvs = [(float(u[0]), 1.0 - float(u[1])) for u in uv_data]
                    uv2_data = _get_accessor_data(getattr(attrs, 'TEXCOORD_1', None))
                    if uv2_data:
                        mesh_node.uvs_lm = [(float(u[0]), 1.0 - float(u[1])) for u in uv2_data]

                    idx_data = _get_accessor_data(prim.indices)
                    if idx_data and len(idx_data) % 3 == 0:
                        mesh_node.faces = [(int(idx_data[i]), int(idx_data[i+1]), int(idx_data[i+2]))
                                           for i in range(0, len(idx_data), 3)]

                    # ── Skin weights (JOINTS_0 / WEIGHTS_0) ──────────────────
                    joints_data  = _get_accessor_data(getattr(attrs, 'JOINTS_0',  None))
                    weights_data = _get_accessor_data(getattr(attrs, 'WEIGHTS_0', None))
                    if joints_data and weights_data and len(joints_data) == len(weights_data):
                        mesh_node.flags = int(NodeFlags.HEADER | NodeFlags.SKIN)
                        skin_data_list = []
                        # Build bone_map from skin joints (resolved later if skin index present)
                        bone_map: List[str] = []
                        skin_idx = gnode.skin
                        if skin_idx is not None and gltf.skins:
                            skin = gltf.skins[skin_idx]
                            for ji in (skin.joints or []):
                                jnode = gltf.nodes[ji]
                                bone_map.append(jnode.name or f"bone_{ji}")
                        mesh_node.bone_map = bone_map

                        for jrow, wrow in zip(joints_data, weights_data):
                            sd = VertexSkinData()
                            sd.influences = []
                            for k in range(4):
                                j_idx = int(jrow[k]) if k < len(jrow) else 0
                                w_val = float(wrow[k]) if k < len(wrow) else 0.0
                                if w_val > 1e-6:
                                    sd.influences.append(BoneWeight(
                                        bone_index=j_idx, weight=w_val))
                            # Normalize weights to sum to 1.0
                            total_w = sum(bw.weight for bw in sd.influences)
                            if total_w > 1e-6:
                                for bw in sd.influences:
                                    bw.weight /= total_w
                            skin_data_list.append(sd)
                        mesh_node.skin_data = skin_data_list

                    # ── Material name + PBR textures ──────────────────────────
                    if prim.material is not None and gltf.materials:
                        mat = gltf.materials[prim.material]
                        if mat.name:
                            mesh_node.texture = mat.name[:32]
                        # Try to get base colour texture name
                        try:
                            pbr = mat.pbrMetallicRoughness
                            if pbr and pbr.baseColorTexture:
                                ti = pbr.baseColorTexture.index
                                tex = gltf.textures[ti]
                                src = gltf.images[tex.source] if tex.source is not None else None
                                if src and src.name:
                                    mesh_node.texture = src.name[:32]
                                elif src and src.uri:
                                    # Strip path + extension to get resref
                                    mesh_node.texture = Path(src.uri).stem[:32]
                        except (AttributeError, TypeError, IndexError):
                            pass

                    mesh_node.render = True
                    mesh_node.has_shadow = True
                    mesh_node.compute_bounds()
                    mnode.children.append(mesh_node)

        # ── Import skeleton / dummy nodes (no mesh) ───────────────────────────
        for i, gnode in enumerate(gltf.nodes or []):
            if gnode.mesh is None and gnode.skin is None:
                # Treat as skeleton/dummy bone
                nm = (gnode.name or f"bone_{i}")[:32]
                # Check if already added as mesh-parent
                existing = next((c for c in root.children if c.name == nm), None)
                if existing is None:
                    tx, ty, tz = 0.0, 0.0, 0.0
                    if gnode.translation:
                        tx, ty, tz = (float(gnode.translation[0]),
                                      float(gnode.translation[1]),
                                      float(gnode.translation[2]))
                    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
                    if gnode.rotation:
                        qx, qy, qz, qw = (float(gnode.rotation[0]),
                                           float(gnode.rotation[1]),
                                           float(gnode.rotation[2]),
                                           float(gnode.rotation[3]))
                    bone_node = ModelNode(
                        name=nm, flags=int(NodeFlags.HEADER),
                        position=(tx, ty, tz), rotation=(qx, qy, qz, qw),
                        parent=root)
                    root.children.append(bone_node)

        # ── Import animations ─────────────────────────────────────────────────
        for ganim in (gltf.animations or []):
            try:
                anim_name = (ganim.name or 'anim')[:32]
                anim_length = 0.0
                anim_nodes_map: Dict[str, ModelNode] = {}

                # Group channels by target node
                from collections import defaultdict
                node_channels: dict = defaultdict(list)
                for ch in (ganim.channels or []):
                    node_channels[ch.target.node].append(ch)

                for target_node_idx, channels in node_channels.items():
                    if target_node_idx is None:
                        continue
                    tnode = gltf.nodes[target_node_idx]
                    tname = (tnode.name or f"node_{target_node_idx}")[:32]
                    anim_mn = ModelNode(name=tname, flags=int(NodeFlags.HEADER))

                    for ch in channels:
                        samp = ganim.samplers[ch.sampler]
                        times_raw = _get_accessor_data(samp.input)
                        values_raw = _get_accessor_data(samp.output)
                        if not times_raw or not values_raw:
                            continue
                        times = [float(t) for t in times_raw]
                        anim_length = max(anim_length, max(times))
                        path = ch.target.path  # 'translation' / 'rotation' / 'scale'
                        if path == 'translation':
                            ctrl_type = 8   # CTRL_POSITION
                            # GLTF position is absolute; KotOR uses delta-from-bind
                            # Store as absolute for now; delta conversion done at play time
                            values = [tuple(float(v) for v in row[:3]) for row in values_raw]
                        elif path == 'rotation':
                            ctrl_type = 20  # CTRL_ORIENTATION
                            values = [tuple(float(v) for v in row[:4]) for row in values_raw]
                        elif path == 'scale':
                            ctrl_type = 36  # CTRL_SCALE
                            values = [(float(row[0]),) for row in values_raw]  # uniform scale
                        else:
                            continue
                        anim_mn.controllers.append({
                            'type': ctrl_type,
                            'times': times,
                            'values': values,
                        })

                    if anim_mn.controllers:
                        anim_nodes_map[tname] = anim_mn

                if anim_nodes_map:
                    anim = Animation()
                    anim.name = anim_name
                    anim.length = anim_length if anim_length > 0 else 1.0
                    anim.nodes  = list(anim_nodes_map.values())
                    model.animations.append(anim)
            except Exception as e:
                anim_name_str = getattr(ganim, "name", "?")
                log.warning(f"GLTFImporter: failed to import animation '{anim_name_str}': {e}")

        model.compute_bounds()
        return model

    def _load_trimesh(self, path, model_name, gv, sm, cl) -> KotorModel:
        import trimesh
        scene = trimesh.load(path)
        model = KotorModel(name=model_name, supermodel=sm, game_version=gv, classification=cl)
        root  = ModelNode(name=model_name, flags=int(NodeFlags.HEADER))
        model.root_node = root
        geoms = scene.geometry if hasattr(scene, 'geometry') else {model_name: scene}
        for gname, mesh in geoms.items():
            n = ModelNode(name=gname[:32], flags=int(NodeFlags.HEADER | NodeFlags.MESH), parent=root)
            n.vertices = [tuple(v) for v in mesh.vertices.tolist()]
            n.faces    = [tuple(f) for f in mesh.faces.tolist()]
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                n.normals = [tuple(x) for x in mesh.vertex_normals.tolist()]
            if (hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv')
                    and mesh.visual.uv is not None):
                n.uvs = [(float(u), 1.0 - float(v)) for u, v in mesh.visual.uv.tolist()]
            n.render = True
            n.has_shadow = True
            n.compute_bounds()
            root.children.append(n)
        model.compute_bounds()
        return model


# ──────────────────────────────────────────────────────────────────────
#  GLTF 2.0 / GLB Exporter
# ──────────────────────────────────────────────────────────────────────

class GLTFExporter:
    """
    Export KotorModel to GLTF 2.0 / GLB.

    Uses pygltflib for structured output. Falls back to manual JSON+binary
    GLTF 2.0 if pygltflib is not available.

    UV convention: KotOR stores V bottom-up; GLTF stores V top-down.
    We flip V on export: v_gltf = 1.0 - v_kotor.
    """

    @staticmethod
    def _node_bind_world_matrix_4x4(node) -> list[list[float]]:
        """Build a 4x4 world bind matrix from node.world_transform().

        Returns a row-major 4x4 matrix (rotation+translation, uniform scale=1).
        """
        import math as _math
        wp, wq = node.world_transform()
        x, y, z, w = float(wq[0]), float(wq[1]), float(wq[2]), float(wq[3])
        # Quaternion to rotation matrix
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 0.0],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0],
            [float(wp[0]), float(wp[1]), float(wp[2]), 1.0],
        ]

    @staticmethod
    def _invert_4x4(m: list[list[float]]) -> list[list[float]]:
        """Invert a row-major 4x4 matrix via Gauss-Jordan elimination."""
        # Build augmented matrix [m | I]
        aug = [list(row[:]) + [1.0 if i == j else 0.0 for j in range(4)] for i, row in enumerate(m)]
        for col in range(4):
            # Find pivot
            pivot = col
            for row in range(col + 1, 4):
                if abs(aug[row][col]) > abs(aug[pivot][col]):
                    pivot = row
            aug[col], aug[pivot] = aug[pivot], aug[col]
            pv = aug[col][col]
            if abs(pv) < 1e-12:
                # Singular — return identity as fallback
                return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
            for j in range(8):
                aug[col][j] /= pv
            for row in range(4):
                if row == col:
                    continue
                factor = aug[row][col]
                for j in range(8):
                    aug[row][j] -= factor * aug[col][j]
        return [row[4:] for row in aug]

    def export(self, model: KotorModel, path: str,
               binary: bool = True, tex_cache=None,
               export_rigging: bool = True) -> bool:
        """
        Export model to GLB (binary=True) or GLTF JSON (binary=False).

        Parameters
        ----------
        model          : KotorModel to export.
        path           : Output file path (.glb or .gltf).
        binary         : True → GLB container; False → GLTF JSON + .bin.
        tex_cache      : Optional TextureCache.  When provided, textures are embedded
                         as base64 data URIs in the GLTF material definitions, so the
                         exported file is self-contained and opens correctly in Blender.
        export_rigging : When True (default) a 'rigging/' subdirectory is created
                         next to the GLB/GLTF file with JSON rigging + animation data
                         in addition to the embedded skeleton in the GLTF.
        Returns True on success.
        """
        out_dir = Path(path).parent
        ok = False
        try:
            ok = self._export_pygltflib(model, path, binary, tex_cache=tex_cache)
        except ImportError:
            pass
        if not ok:
            try:
                ok = self._export_manual(model, path, binary, tex_cache=tex_cache)
            except Exception as e:
                log.error(f"GLTF export failed: {e}")
                return False

        if ok and export_rigging:
            rig_count = _export_rigging_data(model, out_dir)
            if rig_count > 0:
                log.info(f"Rigging data exported: {rig_count} file(s) → "
                         f"{out_dir / 'rigging'}")
        return ok

    # ── pygltflib path ────────────────────────────────────────────────

    @staticmethod
    def _tex_to_base64_uri(tex_cache, tex_name: str) -> Optional[str]:
        """
        Convert a texture from the cache to a data URI (PNG base64).
        Returns None if the texture cannot be found or converted.
        Used to embed textures in GLTF materials for self-contained exports.
        """
        if tex_cache is None or not tex_name:
            return None
        try:
            import io, base64
            img = tex_cache.get(tex_name)
            if img is None:
                return None
            buf = io.BytesIO()
            img.convert('RGBA').save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')
            return f"data:image/png;base64,{b64}"
        except Exception:
            return None

    def _export_pygltflib(self, model: KotorModel, path: str, binary: bool,
                          tex_cache=None) -> bool:
        import pygltflib
        import struct as st
        import base64

        gltf = pygltflib.GLTF2()
        gltf.asset = pygltflib.Asset(version="2.0",
                                     generator="GhostRigger-K1-K2")
        scene = pygltflib.Scene(name=model.name, nodes=[])
        gltf.scenes.append(scene)
        gltf.scene = 0

        bin_data = bytearray()

        def _add_accessor(data_bytes: bytes, count: int, type_: str,
                          component_type: int,
                          min_vals=None, max_vals=None,
                          target: int = None) -> int:
            offset = len(bin_data)
            bin_data.extend(data_bytes)
            # Pad to 4-byte alignment
            while len(bin_data) % 4:
                bin_data.append(0)
            bv_kwargs = dict(
                buffer=0,
                byteOffset=offset,
                byteLength=len(data_bytes),
            )
            if target is not None:
                bv_kwargs["target"] = target
            bv = pygltflib.BufferView(**bv_kwargs)
            bv_idx = len(gltf.bufferViews)
            gltf.bufferViews.append(bv)
            acc = pygltflib.Accessor(
                bufferView=bv_idx,
                byteOffset=0,
                componentType=component_type,
                count=count,
                type=type_,
            )
            if min_vals is not None:
                acc.min = list(min_vals)
            if max_vals is not None:
                acc.max = list(max_vals)
            acc_idx = len(gltf.accessors)
            gltf.accessors.append(acc)
            return acc_idx

        # ── Build skeleton node map (all model nodes as GLTF nodes) ──────────
        # GLTF 2.0 §3.8: skin references a list of joint nodes.
        # We add ALL model nodes (dummy + mesh) to gltf.nodes so that
        # the skeleton hierarchy is intact, then add a skin referencing joints.
        all_model_nodes = list(model.all_nodes()) if model.root_node else []
        gltf_node_idx: Dict[str, int] = {}   # model-node-name → gltf node index

        for mn in all_model_nodes:
            pos_ = list(mn.position) if hasattr(mn, 'position') else [0.0, 0.0, 0.0]
            rot_ = list(mn.rotation) if hasattr(mn, 'rotation') else [0.0, 0.0, 0.0, 1.0]
            skel_node = pygltflib.Node(
                name=mn.name,
                translation=pos_,
                rotation=rot_,
            )
            idx_ = len(gltf.nodes)
            gltf.nodes.append(skel_node)
            gltf_node_idx[mn.name] = idx_
            # Also register lowercase for case-insensitive bone_map lookups
            gltf_node_idx[mn.name.lower()] = idx_

        # Wire parent→child relationships in the GLTF node hierarchy
        for mn in all_model_nodes:
            if mn.parent and mn.parent.name in gltf_node_idx:
                parent_gltf_idx = gltf_node_idx[mn.parent.name]
                child_gltf_idx  = gltf_node_idx[mn.name]
                parent_gltf_node = gltf.nodes[parent_gltf_idx]
                if parent_gltf_node.children is None:
                    parent_gltf_node.children = []
                parent_gltf_node.children.append(child_gltf_idx)
            elif mn.parent is None and mn.name in gltf_node_idx:
                # Root node → add to scene
                scene.nodes.append(gltf_node_idx[mn.name])

        # Build a single skin encompassing all dummy-node joints
        joint_indices = [gltf_node_idx[mn.name]
                         for mn in all_model_nodes
                         if mn.name in gltf_node_idx]
        gltf_skin = None
        if joint_indices:
            gltf_skin_obj = pygltflib.Skin(name=model.name + "_skin", joints=joint_indices)

            # Compute inverse bind matrices for each joint (glTF 2.0 §5.26).
            # Without these, Blender/Unreal import the armature + weights but
            # the bind pose is undefined and deformation is wrong.
            ibm_buf = bytearray()
            for mn in all_model_nodes:
                if mn.name not in gltf_node_idx:
                    continue
                gni = gltf_node_idx[mn.name]
                if gni not in joint_indices:
                    continue
                # Build the 4x4 world bind matrix for this bone
                bone_matrix = self._node_bind_world_matrix_4x4(mn)
                # Invert it
                inv = self._invert_4x4(bone_matrix)
                # glTF stores inverseBindMatrices as column-major MAT4
                for col in range(4):
                    for row in range(4):
                        ibm_buf += st.pack('<f', float(inv[col][row]))
            if ibm_buf and len(joint_indices) > 0:
                ibm_acc = _add_accessor(
                    bytes(ibm_buf), len(joint_indices), "MAT4", 5126,
                )
                gltf_skin_obj.inverseBindMatrices = ibm_acc

            skin_idx = len(gltf.skins)
            gltf.skins.append(gltf_skin_obj)
            gltf_skin = skin_idx

        nodes = _iter_visible_mesh_nodes(model)
        for node in nodes:
            verts   = node.vertices or []
            normals = node.normals  or []
            uvs     = node.uvs     or []
            faces   = node.faces   or []
            if not verts or not faces:
                continue

            n_v = len(verts)
            n_f = len(faces)

            # Position buffer (float32, VEC3)
            pos_buf = bytearray()
            for v in verts:
                pos_buf += st.pack('<fff', float(v[0]), float(v[1]), float(v[2]))
            px = [v[0] for v in verts]; py = [v[1] for v in verts]; pz = [v[2] for v in verts]
            pos_acc = _add_accessor(bytes(pos_buf), n_v, "VEC3", 5126,
                                    [min(px), min(py), min(pz)],
                                    [max(px), max(py), max(pz)],
                                    target=34962)  # ARRAY_BUFFER

            # Normal buffer (float32, VEC3)
            norm_acc = None
            if len(normals) == n_v:
                norm_buf = bytearray()
                for n in normals:
                    norm_buf += st.pack('<fff', float(n[0]), float(n[1]), float(n[2]))
                norm_acc = _add_accessor(bytes(norm_buf), n_v, "VEC3", 5126)

            # UV buffer (float32, VEC2) — flip V for GLTF
            uv_acc = None
            if len(uvs) == n_v:
                uv_buf = bytearray()
                for u in uvs:
                    uv_buf += st.pack('<ff', float(u[0]), 1.0 - float(u[1]))
                uv_acc = _add_accessor(bytes(uv_buf), n_v, "VEC2", 5126)

            # Index buffer (uint32, SCALAR)
            idx_buf = bytearray()
            for f in faces:
                idx_buf += st.pack('<III', int(f[0]), int(f[1]), int(f[2]))
            idx_acc = _add_accessor(bytes(idx_buf), n_f * 3, "SCALAR", 5125,
                                    target=34963)  # ELEMENT_ARRAY_BUFFER

            # Primitive attributes
            prim_attrs = pygltflib.Attributes(POSITION=pos_acc)
            if norm_acc is not None:
                prim_attrs.NORMAL = norm_acc
            if uv_acc is not None:
                prim_attrs.TEXCOORD_0 = uv_acc

            # Skin weights (JOINTS_0 + WEIGHTS_0) for skin nodes
            # GLTF 2.0 §3.7.2: up to 4 influences per vertex as UNSIGNED_BYTE (5121)
            # joints and FLOAT (5126) weights.
            #
            # IMPORTANT: JOINTS_0 values must be indices into skin.joints[], NOT
            # global gltf.nodes[] indices.  KotOR bone_map[bone_index] gives the
            # bone name; we look that name up in gltf_node_idx to get the GLTF
            # node index, then find its position inside joint_indices (the skin's
            # joint list).  Build a local lookup: bone_map_index → joint_list_pos.
            joints_acc = weights_acc = None
            if getattr(node, 'is_skin', False):
                skin_data = getattr(node, 'skin_data', None) or []
                bone_map  = getattr(node, 'bone_map', []) or []
                if skin_data and len(skin_data) == n_v and joint_indices:
                    # Build bone_map_idx → joint_list_position lookup
                    # joint_indices[i] = gltf_node_index for joint i
                    _gltf_node_to_joint_pos = {gni: jpos
                                               for jpos, gni in enumerate(joint_indices)}
                    _bmap_to_joint: Dict[int, int] = {}
                    for bi, bname in enumerate(bone_map):
                        bname_l = bname.lower() if bname else ''
                        # Find the gltf node index for this bone
                        gni = gltf_node_idx.get(bname_l) or gltf_node_idx.get(bname)
                        if gni is not None:
                            _bmap_to_joint[bi] = _gltf_node_to_joint_pos.get(gni, 0)
                        else:
                            _bmap_to_joint[bi] = 0  # fallback: root joint

                    joints_buf  = bytearray()
                    weights_buf = bytearray()
                    for sd in skin_data:
                        infl = sd.influences if hasattr(sd, 'influences') else []
                        # Gather up to 4 influences, sorted by weight descending
                        infl_sorted = sorted(infl, key=lambda x: x.weight, reverse=True)[:4]
                        js = [0, 0, 0, 0]
                        ws = [0.0, 0.0, 0.0, 0.0]
                        for ci, inf in enumerate(infl_sorted):
                            # Remap from bone_map index → skin.joints[] position
                            js[ci] = _bmap_to_joint.get(int(inf.bone_index), 0)
                            ws[ci] = float(inf.weight)
                        # Normalize weights to sum=1 (avoid precision drift)
                        w_sum = sum(ws)
                        if w_sum > 1e-6:
                            ws = [x / w_sum for x in ws]
                        joints_buf  += st.pack('<BBBB', *js)
                        weights_buf += st.pack('<ffff', *ws)
                    # Use UNSIGNED_SHORT (5123) for >255 joints, UNSIGNED_BYTE (5121) otherwise
                    use_short_joints = len(joint_indices) > 255
                    joint_fmt = '<HHHH' if use_short_joints else '<BBBB'
                    joint_type = 5123 if use_short_joints else 5121
                    joints_buf  = bytearray()
                    weights_buf = bytearray()
                    for sd in skin_data:
                        infl = sd.influences if hasattr(sd, 'influences') else []
                        infl_sorted = sorted(infl, key=lambda x: x.weight, reverse=True)[:4]
                        js = [0, 0, 0, 0]
                        ws = [0.0, 0.0, 0.0, 0.0]
                        for ci, inf in enumerate(infl_sorted):
                            js[ci] = _bmap_to_joint.get(int(inf.bone_index), 0)
                            ws[ci] = float(inf.weight)
                        w_sum = sum(ws)
                        if w_sum > 1e-6:
                            ws = [x / w_sum for x in ws]
                        joints_buf  += st.pack(joint_fmt, *js)
                        weights_buf += st.pack('<ffff', *ws)
                    joints_acc  = _add_accessor(bytes(joints_buf),  n_v, "VEC4", joint_type)
                    weights_acc = _add_accessor(bytes(weights_buf), n_v, "VEC4", 5126)
            if joints_acc is not None:
                prim_attrs.JOINTS_0  = joints_acc
            if weights_acc is not None:
                prim_attrs.WEIGHTS_0 = weights_acc

            # Material (with optional embedded texture)
            mat_idx = None
            tex_name = str(getattr(node, 'texture_clean', '') or
                           getattr(node, 'texture', '') or '').strip()
            if tex_name and tex_name.upper() not in ('NULL', 'BLACK', ''):
                mat = pygltflib.Material(name=tex_name, doubleSided=False)
                diff = getattr(node, 'diffuse', (1.0, 1.0, 1.0))
                alpha_val = float(getattr(node, 'alpha', 1.0))
                pbr = pygltflib.PbrMetallicRoughness(
                    baseColorFactor=[float(diff[0]), float(diff[1]),
                                     float(diff[2]), alpha_val],
                    metallicFactor=0.0,
                    roughnessFactor=0.8,
                )
                # Embed texture as base64 PNG data URI if tex_cache available
                uri = self._tex_to_base64_uri(tex_cache, tex_name)
                if uri is not None:
                    img_obj = pygltflib.Image(uri=uri, name=tex_name)
                    img_idx = len(gltf.images)
                    gltf.images.append(img_obj)
                    sampler = pygltflib.Sampler(
                        magFilter=9729, minFilter=9987,  # LINEAR, LINEAR_MIPMAP_LINEAR
                        wrapS=10497, wrapT=10497,        # REPEAT
                    )
                    samp_idx = len(gltf.samplers)
                    gltf.samplers.append(sampler)
                    tex_obj = pygltflib.Texture(source=img_idx, sampler=samp_idx,
                                                name=tex_name)
                    tex_idx = len(gltf.textures)
                    gltf.textures.append(tex_obj)
                    pbr.baseColorTexture = pygltflib.TextureInfo(index=tex_idx)
                mat.pbrMetallicRoughness = pbr
                if alpha_val < 0.999:
                    mat.alphaMode = "BLEND"
                mat_idx = len(gltf.materials)
                gltf.materials.append(mat)

            prim = pygltflib.Primitive(
                attributes=prim_attrs,
                indices=idx_acc,
                material=mat_idx,
            )
            gmesh = pygltflib.Mesh(name=node.name, primitives=[prim])
            mesh_idx = len(gltf.meshes)
            gltf.meshes.append(gmesh)

            # Attach mesh to the pre-built skeleton node if it exists,
            # otherwise create a standalone mesh node
            if node.name in gltf_node_idx:
                existing_gnode = gltf.nodes[gltf_node_idx[node.name]]
                existing_gnode.mesh = mesh_idx
                # Assign skin for skinned meshes
                if joints_acc is not None and gltf_skin is not None:
                    existing_gnode.skin = gltf_skin
            else:
                pos = node.position or (0.0, 0.0, 0.0)
                rot = node.rotation or (0.0, 0.0, 0.0, 1.0)
                gnode = pygltflib.Node(
                    name=node.name,
                    mesh=mesh_idx,
                    translation=list(pos),
                    rotation=list(rot),  # GLTF: [x, y, z, w]
                    skin=gltf_skin if (joints_acc is not None) else None,
                )
                node_idx = len(gltf.nodes)
                gltf.nodes.append(gnode)
                scene.nodes.append(node_idx)

        # ── Export animations ─────────────────────────────────────────────────
        for anim in (model.animations or []):
            try:
                ganim = pygltflib.Animation(name=anim.name)
                ganim.channels = []
                ganim.samplers = []
                sampler_idx = 0

                for anim_node in (anim.nodes or []):
                    tgt_node_idx = gltf_node_idx.get(anim_node.name) or \
                                   gltf_node_idx.get(anim_node.name.lower())
                    if tgt_node_idx is None:
                        continue

                    raw_controllers = anim_node.controllers or []
                    for ctrl in raw_controllers:
                        # Normalize legacy dict-keyed controllers to list format
                        if isinstance(ctrl, int):
                            legacy = raw_controllers.get(ctrl) if isinstance(raw_controllers, dict) else None
                            if not isinstance(legacy, dict):
                                continue
                            ctrl = legacy
                        if not isinstance(ctrl, dict):
                            continue
                        ctype  = ctrl.get('type')
                        times  = ctrl.get('times', [])
                        values = ctrl.get('values', [])
                        if not times or not values:
                            continue

                        # Map controller type → GLTF path + type string
                        if ctype == 8:    # CTRL_POSITION → translation
                            path_str = 'translation'
                            val_type = 'VEC3'
                            val_fmt  = '<fff'
                            val_size = 12
                            def _pack_v(v):
                                return st.pack('<fff', float(v[0]), float(v[1]), float(v[2]))
                        elif ctype == 20: # CTRL_ORIENTATION → rotation
                            path_str = 'rotation'
                            val_type = 'VEC4'
                            val_fmt  = '<ffff'
                            val_size = 16
                            def _pack_v(v):
                                return st.pack('<ffff', float(v[0]), float(v[1]),
                                               float(v[2]), float(v[3]))
                        elif ctype == 36: # CTRL_SCALE → scale
                            path_str = 'scale'
                            val_type = 'VEC3'
                            val_fmt  = '<fff'
                            val_size = 12
                            def _pack_v(v):
                                sv = float(v[0]) if hasattr(v, '__len__') else float(v)
                                return st.pack('<fff', sv, sv, sv)
                        else:
                            continue

                        # Time accessor
                        t_buf = bytearray()
                        for t in times:
                            t_buf += st.pack('<f', float(t))
                        t_acc = _add_accessor(bytes(t_buf), len(times), 'SCALAR', 5126,
                                              [min(times)], [max(times)])

                        # Value accessor
                        v_buf = bytearray()
                        for v in values:
                            v_buf += _pack_v(v)
                        v_acc = _add_accessor(bytes(v_buf), len(values), val_type, 5126)

                        samp = pygltflib.AnimationSampler(input=t_acc, output=v_acc,
                                                          interpolation='LINEAR')
                        ganim.samplers.append(samp)
                        ch = pygltflib.AnimationChannel(
                            sampler=sampler_idx,
                            target=pygltflib.AnimationChannelTarget(
                                node=tgt_node_idx, path=path_str)
                        )
                        ganim.channels.append(ch)
                        sampler_idx += 1

                if ganim.channels:
                    gltf.animations.append(ganim)
            except Exception as e:
                log.warning(f"GLTF export: failed to export animation '{anim.name}': {e}")

        # Finalize buffer
        if binary or path.lower().endswith('.glb'):
            # GLB: binary blob is embedded; buffer must NOT have a uri
            gltf.buffers.append(pygltflib.Buffer(byteLength=len(bin_data)))
            gltf.set_binary_blob(bytes(bin_data))
            out_path = path if path.lower().endswith('.glb') else str(Path(path).with_suffix('.glb'))
            gltf.save(out_path)
            log.info(f"GLB export → {Path(out_path).name}")
        else:
            # GLTF JSON: write binary to a sidecar .bin file with a uri
            bin_uri = Path(path).with_suffix('.bin').name
            gltf.buffers.append(pygltflib.Buffer(byteLength=len(bin_data), uri=bin_uri))
            bin_path = str(Path(path).with_suffix('.bin'))
            Path(bin_path).write_bytes(bytes(bin_data))
            gltf.save(path)
            log.info(f"GLTF export → {Path(path).name} (+ {Path(bin_path).name})")
        return True

    # ── Manual GLTF JSON+BIN path (no external deps) ─────────────────

    def _export_manual(self, model: KotorModel, path: str, binary: bool,
                       tex_cache=None) -> bool:
        """
        Write a minimal GLTF 2.0 file without external dependencies.
        Outputs .gltf + embedded base64 buffer or .glb binary container.
        When tex_cache is supplied, textures are embedded as PNG base64 data URIs.
        """
        import json, struct as st, base64

        buffers_bytes = bytearray()
        accessors   = []
        buffer_views = []
        meshes      = []
        nodes_list  = []
        materials   = []
        scene_nodes = []

        def _add_bv(data: bytes) -> int:
            off = len(buffers_bytes)
            buffers_bytes.extend(data)
            while len(buffers_bytes) % 4:
                buffers_bytes.append(0)
            bv = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
            idx = len(buffer_views)
            buffer_views.append(bv)
            return idx

        def _add_acc(data: bytes, count: int, typ: str, comp: int,
                     min_v=None, max_v=None) -> int:
            bv_i = _add_bv(data)
            acc  = {"bufferView": bv_i, "byteOffset": 0,
                    "componentType": comp, "count": count, "type": typ}
            if min_v: acc["min"] = list(min_v)
            if max_v: acc["max"] = list(max_v)
            idx = len(accessors)
            accessors.append(acc)
            return idx

        for node in _iter_visible_mesh_nodes(model):
            verts  = node.vertices or []
            norms  = node.normals  or []
            uvs    = node.uvs      or []
            faces  = node.faces    or []
            if not verts or not faces:
                continue
            nv = len(verts)
            nf = len(faces)

            pb  = bytearray()
            for v in verts: pb += st.pack('<fff', *[float(x) for x in v])
            px = [v[0] for v in verts]; py = [v[1] for v in verts]; pz = [v[2] for v in verts]
            pa = _add_acc(bytes(pb), nv, "VEC3", 5126,
                          [min(px), min(py), min(pz)], [max(px), max(py), max(pz)])

            na = None
            if len(norms) == nv:
                nb = bytearray()
                for n in norms: nb += st.pack('<fff', *[float(x) for x in n])
                na = _add_acc(bytes(nb), nv, "VEC3", 5126)

            ua = None
            if len(uvs) == nv:
                ub = bytearray()
                for u in uvs: ub += st.pack('<ff', float(u[0]), 1.0 - float(u[1]))
                ua = _add_acc(bytes(ub), nv, "VEC2", 5126)

            ib  = bytearray()
            for f in faces: ib += st.pack('<III', *[int(x) for x in f[:3]])
            ia  = _add_acc(bytes(ib), nf * 3, "SCALAR", 5125)

            attrs = {"POSITION": pa}
            if na is not None: attrs["NORMAL"] = na
            if ua is not None: attrs["TEXCOORD_0"] = ua

            mat_i = None
            tex   = str(getattr(node, 'texture_clean', '') or
                        getattr(node, 'texture', '') or '').strip()
            if tex and tex.upper() not in ('NULL', 'BLACK', ''):
                diff      = getattr(node, 'diffuse', (1.0, 1.0, 1.0))
                alpha_val = float(getattr(node, 'alpha', 1.0))
                pbr_dict  = {
                    "baseColorFactor": [float(diff[0]), float(diff[1]),
                                        float(diff[2]), alpha_val],
                    "metallicFactor": 0.0, "roughnessFactor": 0.8
                }
                mat_obj: dict = {"name": tex, "pbrMetallicRoughness": pbr_dict}
                if alpha_val < 0.999:
                    mat_obj["alphaMode"] = "BLEND"
                mat_i = len(materials)
                materials.append(mat_obj)

            prim = {"attributes": attrs, "indices": ia}
            if mat_i is not None: prim["material"] = mat_i
            meshes.append({"name": node.name, "primitives": [prim]})
            mi = len(meshes) - 1

            pos = list(node.position or (0.0, 0.0, 0.0))
            rot = list(node.rotation or (0.0, 0.0, 0.0, 1.0))
            ni = len(nodes_list)
            nodes_list.append({"name": node.name, "mesh": mi,
                                "translation": pos, "rotation": rot})
            scene_nodes.append(ni)

        buf_b64 = base64.b64encode(bytes(buffers_bytes)).decode('ascii')

        # ── Embed textures as PNG data URIs ──────────────────────────────────
        # Build images / samplers / textures arrays for any material that has a
        # known texture name, then patch the material's baseColorTexture reference.
        gltf_images:   List[dict] = []
        gltf_samplers: List[dict] = []
        gltf_textures: List[dict] = []
        _tex_name_to_idx: Dict[str, int] = {}  # tex_name → gltf_textures index

        if tex_cache is not None and materials:
            for mat in materials:
                tex_n = mat.get("name", "")
                if not tex_n or tex_n in _tex_name_to_idx:
                    if tex_n in _tex_name_to_idx:
                        # Patch this material too
                        mat["pbrMetallicRoughness"]["baseColorTexture"] = {
                            "index": _tex_name_to_idx[tex_n]}
                    continue
                uri = self._tex_to_base64_uri(tex_cache, tex_n)
                if uri is None:
                    continue
                img_idx  = len(gltf_images)
                samp_idx = len(gltf_samplers)
                tex_idx  = len(gltf_textures)
                gltf_images.append({"uri": uri, "name": tex_n})
                gltf_samplers.append({"magFilter": 9729, "minFilter": 9987,
                                      "wrapS": 10497, "wrapT": 10497})
                gltf_textures.append({"source": img_idx, "sampler": samp_idx,
                                      "name": tex_n})
                _tex_name_to_idx[tex_n] = tex_idx
                mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": tex_idx}

        # ── Build node index for animation channel targeting ──────────────────
        node_name_to_idx: Dict[str, int] = {
            n.get("name", ""): i for i, n in enumerate(nodes_list)
        }

        # ── Export animations (manual path) ───────────────────────────────────
        gltf_animations: List[dict] = []
        for anim in (model.animations or []):
            try:
                anim_samplers: List[dict] = []
                anim_channels: List[dict] = []
                samp_i = 0
                for anim_node in (anim.nodes or []):
                    tgt_idx = node_name_to_idx.get(anim_node.name)
                    if tgt_idx is None:
                        tgt_idx = node_name_to_idx.get(anim_node.name.lower())
                    if tgt_idx is None:
                        continue
                    for ctrl in (anim_node.controllers or []):
                        ctype  = ctrl.get('type')
                        times  = ctrl.get('times', [])
                        values = ctrl.get('values', [])
                        if not times or not values:
                            continue
                        if ctype == 8:
                            path_str, val_type = 'translation', 'VEC3'
                            def _pv(v): return st.pack('<fff', float(v[0]), float(v[1]), float(v[2]))
                        elif ctype == 20:
                            path_str, val_type = 'rotation', 'VEC4'
                            def _pv(v): return st.pack('<ffff', float(v[0]), float(v[1]),
                                                       float(v[2]), float(v[3]))
                        elif ctype == 36:
                            path_str, val_type = 'scale', 'VEC3'
                            def _pv(v):
                                sv = float(v[0]) if hasattr(v, '__len__') else float(v)
                                return st.pack('<fff', sv, sv, sv)
                        else:
                            continue
                        t_b = bytearray()
                        for t in times: t_b += st.pack('<f', float(t))
                        v_b = bytearray()
                        for v in values: v_b += _pv(v)
                        t_acc = _add_acc(bytes(t_b), len(times), 'SCALAR', 5126,
                                         [min(times)], [max(times)])
                        v_acc = _add_acc(bytes(v_b), len(values), val_type, 5126)
                        anim_samplers.append({
                            "input": t_acc, "output": v_acc, "interpolation": "LINEAR"})
                        anim_channels.append({
                            "sampler": samp_i,
                            "target": {"node": tgt_idx, "path": path_str}})
                        samp_i += 1
                if anim_channels:
                    gltf_animations.append({
                        "name": anim.name,
                        "samplers": anim_samplers,
                        "channels": anim_channels,
                    })
            except Exception as e:
                log.warning(f"GLTF manual export: anim '{getattr(anim,'name','?')}': {e}")

        buf_b64 = base64.b64encode(bytes(buffers_bytes)).decode('ascii')

        gltf_json = {
            "asset": {"version": "2.0", "generator": "GhostRigger-K1-K2"},
            "scene": 0,
            "scenes": [{"name": model.name, "nodes": scene_nodes}],
            "nodes": nodes_list,
            "meshes": meshes,
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [{"byteLength": len(buffers_bytes),
                         "uri": "data:application/octet-stream;base64," + buf_b64}],
        }
        if materials:
            gltf_json["materials"] = materials
        if gltf_images:
            gltf_json["images"]   = gltf_images
            gltf_json["samplers"] = gltf_samplers
            gltf_json["textures"] = gltf_textures
        if gltf_animations:
            gltf_json["animations"] = gltf_animations

        out_path = path
        if binary or path.endswith('.glb'):
            # Write GLB container
            if not out_path.endswith('.glb'):
                out_path = str(Path(path).with_suffix('.glb'))
            json_bytes = json.dumps(gltf_json).encode('utf-8')
            while len(json_bytes) % 4:
                json_bytes += b' '
            buf_bytes = bytes(buffers_bytes)
            while len(buf_bytes) % 4:
                buf_bytes += b'\x00'
            total = 12 + 8 + len(json_bytes) + 8 + len(buf_bytes)
            with open(out_path, 'wb') as f:
                f.write(st.pack('<III', 0x46546C67, 2, total))  # magic, version, length
                f.write(st.pack('<II', len(json_bytes), 0x4E4F534A))  # chunk len, JSON
                f.write(json_bytes)
                f.write(st.pack('<II', len(buf_bytes), 0x004E4942))   # chunk len, BIN
                f.write(buf_bytes)
        else:
            # Remove embedded buffer URI for external bin file
            bin_path = str(Path(path).with_suffix('.bin'))
            gltf_json["buffers"][0]["uri"] = Path(bin_path).name
            with open(bin_path, 'wb') as f:
                f.write(bytes(buffers_bytes))
            with open(path, 'w') as f:
                json.dump(gltf_json, f, indent=2)

        log.info(f"GLTF manual export → {Path(out_path).name}")
        return True


# ──────────────────────────────────────────────────────────────────────
#  glTF Round-Trip Verification Helpers
# ──────────────────────────────────────────────────────────────────────

class GltfRoundTripResult:
    """Result object returned by :func:`gltf_round_trip_verify`.

    Attributes:
        ok (bool): True if the round-trip passed all checks.
        mesh_count_match (bool): Export and import mesh counts agree.
        vertex_count_delta (dict): {mesh_name: (export_count, import_count)}.
        face_count_delta (dict):   {mesh_name: (export_count, import_count)}.
        node_names_missing (list): Bone/node names present in original but lost.
        node_names_extra (list):   Extra nodes introduced by the round-trip.
        animation_names_ok (bool): All animation names survived.
        animation_names_missing (list): Animation names lost in round-trip.
        uv_max_delta (float): Maximum UV coordinate difference (0.0 = perfect).
        errors (list[str]): List of error/warning messages.
    """

    def __init__(self):
        self.ok = False
        self.mesh_count_match = False
        self.vertex_count_delta: Dict[str, tuple] = {}
        self.face_count_delta:   Dict[str, tuple] = {}
        self.node_names_missing: List[str] = []
        self.node_names_extra:   List[str] = []
        self.animation_names_ok  = False
        self.animation_names_missing: List[str] = []
        self.uv_max_delta = 0.0
        self.errors: List[str] = []

    def summary(self) -> str:
        lines = [f"Round-trip OK={self.ok}"]
        lines.append(f"  mesh_count_match={self.mesh_count_match}")
        for name, (exp, imp) in self.vertex_count_delta.items():
            lines.append(f"  vertex {name}: export={exp} import={imp} delta={imp-exp:+d}")
        for name, (exp, imp) in self.face_count_delta.items():
            lines.append(f"  faces  {name}: export={exp} import={imp} delta={imp-exp:+d}")
        if self.node_names_missing:
            lines.append(f"  nodes missing: {self.node_names_missing}")
        if self.node_names_extra:
            lines.append(f"  nodes extra:   {self.node_names_extra}")
        lines.append(f"  anim_names_ok={self.animation_names_ok}")
        if self.animation_names_missing:
            lines.append(f"  anims missing: {self.animation_names_missing}")
        lines.append(f"  uv_max_delta={self.uv_max_delta:.6f}")
        if self.errors:
            lines.append(f"  errors: {self.errors}")
        return "\n".join(lines)


def gltf_round_trip_verify(
    model: 'KotorModel',
    *,
    binary: bool = True,
    tmp_dir: Optional[str] = None,
    uv_tolerance: float = 1e-4,
) -> 'GltfRoundTripResult':
    """Export *model* to a temporary glTF file and immediately re-import it.

    Compares the original and re-imported models for:
      - Mesh count agreement
      - Per-mesh vertex and face count agreement (± rounding from triangulation)
      - Node/bone name preservation
      - Animation name preservation
      - UV coordinate round-trip accuracy

    Args:
        model:         The source :class:`KotorModel` to test.
        binary:        If True, write GLB; if False, write GLTF+BIN.
        tmp_dir:       Directory for temp files (``None`` → :mod:`tempfile` default).
        uv_tolerance:  Maximum acceptable UV delta before the UV check is flagged.

    Returns:
        :class:`GltfRoundTripResult` with detailed pass/fail information.
    """
    import tempfile, os as _os

    result = GltfRoundTripResult()
    ext    = '.glb' if binary else '.gltf'
    exporter = GLTFExporter()
    importer = GLTFImporter()

    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        out_path = _os.path.join(td, f"rtrip_verify{ext}")

        # ── Export ────────────────────────────────────────────────────────
        try:
            ok = exporter.export(model, out_path, binary=binary)
            if not ok:
                result.errors.append("GLTFExporter.export() returned False")
                return result
        except Exception as exc:
            result.errors.append(f"Export failed: {exc}")
            return result

        # ── Import ────────────────────────────────────────────────────────
        try:
            rt_model = importer.import_file(out_path)
            if rt_model is None:
                result.errors.append("GLTFImporter.import_file() returned None")
                return result
        except Exception as exc:
            result.errors.append(f"Import failed: {exc}")
            return result

    # ── Compare mesh nodes ────────────────────────────────────────────────
    orig_mesh_nodes = [n for n in model.all_nodes() if n.vertices]
    rt_mesh_nodes   = [n for n in rt_model.all_nodes() if n.vertices]
    result.mesh_count_match = len(orig_mesh_nodes) == len(rt_mesh_nodes)
    if not result.mesh_count_match:
        result.errors.append(
            f"Mesh count mismatch: export={len(orig_mesh_nodes)} import={len(rt_mesh_nodes)}")

    # Per-mesh vertex / face comparison (by name, best-effort)
    rt_by_name: Dict[str, 'ModelNode'] = {n.name.lower(): n for n in rt_mesh_nodes}
    for on in orig_mesh_nodes:
        rt_n = rt_by_name.get(on.name.lower())
        if rt_n is None:
            result.errors.append(f"Mesh '{on.name}' lost in import")
            result.vertex_count_delta[on.name] = (len(on.vertices), 0)
            result.face_count_delta[on.name]   = (len(on.faces), 0)
            continue
        result.vertex_count_delta[on.name] = (len(on.vertices), len(rt_n.vertices))
        result.face_count_delta[on.name]   = (len(on.faces), len(rt_n.faces))

        # UV delta check (compare first min(N,100) UVs)
        if on.uvs and rt_n.uvs:
            n_check = min(len(on.uvs), len(rt_n.uvs), 100)
            for i in range(n_check):
                ou, ov = on.uvs[i]
                ru, rv = rt_n.uvs[i]
                # GLTF V-flip: export flips v = 1-v, import should flip back
                delta = max(abs(ou - ru), abs(ov - rv))
                if delta > result.uv_max_delta:
                    result.uv_max_delta = delta

    if result.uv_max_delta > uv_tolerance:
        result.errors.append(
            f"UV round-trip delta {result.uv_max_delta:.6f} exceeds tolerance {uv_tolerance}")

    # ── Compare node names ────────────────────────────────────────────────
    orig_names = {n.name.lower() for n in model.all_nodes() if n.name}
    rt_names   = {n.name.lower() for n in rt_model.all_nodes() if n.name}
    result.node_names_missing = sorted(orig_names - rt_names)
    result.node_names_extra   = sorted(rt_names   - orig_names)
    # Non-mesh helper nodes (e.g. root 'Scene') commonly appear in imports
    if result.node_names_missing:
        result.errors.append(f"Nodes lost: {result.node_names_missing[:10]}")

    # ── Compare animation names ───────────────────────────────────────────
    orig_anims = {a.name.lower() for a in model.animations}
    rt_anims   = {a.name.lower() for a in rt_model.animations}
    result.animation_names_missing = sorted(orig_anims - rt_anims)
    result.animation_names_ok = (not result.animation_names_missing)
    if result.animation_names_missing:
        result.errors.append(f"Animations lost: {result.animation_names_missing[:10]}")

    # ── Overall pass/fail ─────────────────────────────────────────────────
    result.ok = (
        result.mesh_count_match
        and result.uv_max_delta <= uv_tolerance
        and not result.node_names_missing
        and result.animation_names_ok
        and not result.errors
    )
    return result
