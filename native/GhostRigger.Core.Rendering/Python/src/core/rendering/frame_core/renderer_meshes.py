"""RendererMeshMixin methods for the viewport frame renderer."""

from __future__ import annotations

from .mixin_imports import (
    Image,
    ImageDraw,
    _ACCEL_AVAILABLE,
    _BG,
    _FACE_MESH_SUBSTRINGS,
    _INNER_GEO_SUBSTRINGS,
    _NUMPY,
    _PIL,
    _SEL,
    _WIRE,
    _accel_depth_sort,
    _accel_flat_shade_frame,
    _accel_frustum_cull,
    _accel_rasterize_frame,
    _clamp,
    _clean_tex_name,
    _compute_flipbook_uv,
    _dot,
    _edge_has_seam_global,
    _float_to_sort_key,
    _paste_lightmap_triangle,
    _paste_textured_triangle,
    _uwrap_global,
    is_animation_supermodel,
    log,
    math,
    np,
)
from src.core.rendering.mesh_render_data import (
    _node_has_renderable_rigid_surface,
    _pose_node_for_transform,
)
from src.core.rendering.gpu_diagnostics_records import _node_uses_single_tile_atlas


class RendererMeshMixin:
    def _draw_mesh_flat(self, draw: 'ImageDraw.Draw',
                        img: 'Image.Image', W: int, H: int):
        cam      = self.cam
        # Use pre-cached view matrix from render() to avoid recomputing per call
        right, up, fwd, eye = getattr(self, '_frame_view', None) or cam._view_matrix()

        # UE-inspired: screen-size driven triangle cap.
        # Scales budget between MAX_TRIS_INTERACTIVE and MAX_TRIS based on how
        # large the model appears on screen (like UE's ComputeBoundsScreenSize).
        tri_cap = self._screen_size_lod_cap(W, H)

        tris = []  # (sort_key, screen_pts, fill_rgb, is_selected)

        for node in self._iter_mesh_nodes():
            if not node.vertices or not node.faces:
                continue
            is_sel = (node is self.selected_node)

            # Skip nodes explicitly marked non-renderable (render=False).
            # Respect the KotOR MDL render flag; only bypass for selected node.
            # EXCEPTION: inner-geometry nodes (eyes, eyelids, teeth, tongue, jaw,
            # gum) are ALWAYS rendered even if render=0.  Some KotOR NPC head MDLs
            # incorrectly store render=0 on these nodes; skipping them causes the
            # character to appear eyeless / toothless in the viewport.
            _nl_flat = node.name.lower()
            _is_inner_geo_flat = any(s in _nl_flat for s in _INNER_GEO_SUBSTRINGS)
            if not getattr(node, 'render', True) and not is_sel and not _is_inner_geo_flat:
                continue

            # Skip deformation-helper nodes entirely in flat mode (unless selected).
            # These are internal skinning proxy meshes that only clutter the display.
            is_helper = self._is_deformation_helper(node)
            if is_helper and not is_sel:
                continue
            # Also skip outlier skin proxies (far-body meshes in accessory models)
            if self._is_outlier_skin(node) and not is_sel:
                continue

            verts = node.vertices
            nv    = len(verts)

            # Pre-transform ALL vertices for this node once.
            # Uses LBS (linear blend skinning) when animation pose is active,
            # otherwise falls back to bind-pose transform.
            # PERF-FIX (v10.2): Use per-frame vertex/normal cache.
            _node_id_flat = id(node)
            _fvc_flat = getattr(self, '_frame_verts_cache', None)
            _fnc_flat = getattr(self, '_frame_norms_cache', None)
            if _fvc_flat is not None and _node_id_flat in _fvc_flat:
                world_verts = _fvc_flat[_node_id_flat]
            else:
                world_verts = self._get_world_verts_for_node(node)
                if _fvc_flat is not None:
                    _fvc_flat[_node_id_flat] = world_verts

            # Pre-transform normals to world space for correct lighting on
            # rotated non-skin nodes (e.g. Wardroid / c_brith body panels).
            if _fnc_flat is not None and _node_id_flat in _fnc_flat:
                world_norms = _fnc_flat[_node_id_flat]
            else:
                world_norms = self._get_world_normals_for_node(node)
                if _fnc_flat is not None:
                    _fnc_flat[_node_id_flat] = world_norms
            n_norms = len(world_norms)

            # ── UE-inspired area-weighted vertex normals ──────────────────────
            # If the node has no stored normals (many KotOR placeable/prop nodes
            # omit per-vertex normals in their MDX data), compute them from the
            # face geometry using area-weighted accumulation.  This gives much
            # smoother shading on curved surfaces than per-face flat-normals.
            # Reference: UE5 SkeletalRenderCPUSkin.cpp tangent accumulation loops.
            if n_norms == 0 and world_verts and node.faces:
                world_norms = self._compute_area_weighted_normals(node.faces, world_verts)
                n_norms = len(world_norms)

            # Batch-project all world vertices to screen coords once per node
            # (avoids per-face view-matrix reconstruction — significant speedup)
            screen_verts = self._proj_batch(world_verts, W, H)


            # Base colour: use texture diffuse or grey for untextured
            clean_tex = _clean_tex_name(node.texture)
            if not clean_tex or clean_tex.upper() in ('NULL',''):
                r, g, b = 130, 130, 160
            else:
                r  = int(_clamp(node.diffuse[0] * 220, 30, 240))
                g  = int(_clamp(node.diffuse[1] * 220, 30, 240))
                b  = int(_clamp(node.diffuse[2] * 220, 30, 240))

            if node.is_skin:
                b = min(b + 25, 255)   # slight blue tint for skin nodes

            # ── Bumpmap/envmap visual indicator (flat mode) ──────────────────
            # Since we can't do real normal-mapping in the flat rasteriser, give
            # bumpmap nodes a subtle warm-gold tint and envmap nodes a cyan tint
            # so modders know these surfaces have special material effects.
            _has_bump = bool(getattr(node, 'txi_bumpmaptexture', ''))
            _has_env  = bool(getattr(node, 'txi_envmaptexture', ''))
            if _has_bump:
                r = min(255, int(r * 1.10 + 10))   # warm gold tint
                g = min(255, int(g * 1.05))
            if _has_env:
                g = min(255, g + 15)               # cyan tint
                b = min(255, b + 20)

            # Per-node alpha — transparent nodes (glass, droid eyes) get blended
            node_alpha = float(getattr(node, 'alpha', 1.0))
            node_alpha = _clamp(node_alpha, 0.0, 1.0)
            # transparency_hint is a render-mode flag, NOT an alpha value:
            #   0 = opaque (default), 1 = additive, 2 = subtractive/special.
            # Do NOT force partial alpha from transparency_hint alone —
            # only honour explicit alpha < 1.0 set by CTRL_MESH_ALPHA or node.alpha.

            # Apply animated alpha from pose (CTRL_MESH_ALPHA=132)
            if self._anim_pose is not None:
                _pn_flat = _pose_node_for_transform(node, self._anim_pose)
                if _pn_flat is not None and _pn_flat.alpha is not None:
                    node_alpha = _clamp(_pn_flat.alpha, 0.0, 1.0)

            # ── Cloth/dangly mesh: teal tint to visually distinguish cloth ──
            # Dangly (cloth) nodes get a distinctive teal colour overlay so
            # modders can immediately see which geometry has cloth simulation.
            is_cloth = node.is_dangly
            # Two-sided: cloth/dangly + transparent materials skip backface cull.
            # Also make face/head mesh nodes two-sided: KotOR head models have
            # inner-geometry (eyes, teeth, tongue) sitting INSIDE the face mesh.
            # Backface culling on the face can make interior geometry visible
            # from directions where the face mesh winding appears reversed (e.g.
            # looking upward through the mouth gap, or in some model orientations).
            # Rendering the face as two-sided prevents the "see-through" effect
            # without changing the depth-sort ordering.
            _nl_flat2 = node.name.lower()
            _is_face_mesh_flat = any(s in _nl_flat2 for s in _FACE_MESH_SUBSTRINGS)
            # BUG FIX v26: inner-geometry nodes (eyes, eyelids, teeth, tongue) sit
            # INSIDE the face mesh.  Without two-sided rendering their triangles are
            # back-face culled when viewed from outside the head (the eye normals
            # typically point inward/outward inconsistently).  Force two-sided so
            # they always contribute pixels after the tier-1 promotion draws them
            # over the opaque head mesh.
            _is_inner_geo_flat2 = any(s in _nl_flat2 for s in _INNER_GEO_SUBSTRINGS)
            is_two_sided_flat = (is_cloth
                                 or getattr(node, "transparency_hint", 0) in (1, 2)
                                 or _is_face_mesh_flat
                                 or _is_inner_geo_flat2)
            if is_cloth:
                # Teal shift: boost green+blue, reduce red
                r = max(20, r - 40)
                g = min(255, g + 60)
                b = min(255, b + 80)

            # ── Inner-geometry tier bump (eyes, teeth, eyelids, tongue) ─────
            # In KotOR heads, eye/teeth/eyelid nodes sit geometrically INSIDE the
            # head mesh (behind the eye-socket opening / mouth gap).  They have
            # transparency_hint=0 (opaque) just like the face mesh, so the standard
            # two-pass tier (0=opaque first, 1=transparent last) would lump them
            # together and rely purely on centroid depth to decide draw order.
            # Centroid depth alone fails here: the eyeball centroid may be computed
            # as FURTHER from the camera than the whole-head centroid, causing the
            # head mesh to be drawn LAST and paint over the eyeball.
            # Fix: promote these inner-geometry nodes to tier 1 so they are ALWAYS
            # drawn AFTER the opaque head/body mesh regardless of depth order.
            # The head mesh's geometric eye-socket opening then correctly exposes the
            # eyeball geometry underneath.
            _nl_flat = node.name.lower()
            # Inner-geometry tier bump: promote eye/teeth/tongue nodes to draw tier 1
            # (after opaque face mesh) so they show through socket/mouth openings.
            # BUG FIX v20: removed 'not node.is_skin' gate — in some K2 head models
            # (child_f, comm_a_m, p_carth, etc.) eyeball nodes ARE declared as skin
            # meshes (MESH|SKIN flags) rather than trimesh.  The old check prevented
            # these skin-type eyeballs from being promoted, causing them to be painter-
            # sorted behind the opaque face mesh and become invisible.  Now we check
            # ALL nodes (skin or not) for inner-geo naming, as long as they have a
            # non-null texture (deformation helpers with null textures won't match).
            _clean_tex_flat = _clean_tex_name(getattr(node, 'texture', '') or '')
            _has_tex_flat = bool(_clean_tex_flat and _clean_tex_flat.upper() not in ('NULL', ''))
            _is_inner_geo_flat = (
                _has_tex_flat
                and any(s in _nl_flat for s in _INNER_GEO_SUBSTRINGS)
                and int(getattr(node, 'transparency_hint', 0)) == 0
            )

            for fi, face in enumerate(node.faces):
                if len(face) < 3: continue
                v0, v1, v2 = face[0], face[1], face[2]
                if v0 >= nv or v1 >= nv or v2 >= nv:
                    continue
                # Skip degenerate (collapsed) faces with repeated vertex indices
                if v0 == v1 or v1 == v2 or v0 == v2:
                    continue

                wv0 = world_verts[v0]
                wv1 = world_verts[v1]
                wv2 = world_verts[v2]
                p0 = screen_verts[v0]
                p1 = screen_verts[v1]
                p2 = screen_verts[v2]
                if p0 is None or p1 is None or p2 is None:
                    continue

                # ── Backface culling (screen-space winding order) ────────
                # In screen space (Y-axis pointing DOWN), the cross product
                # sign convention is: winding < 0 → CCW (front-facing in
                # right-handed KotOR coordinates).  winding > 0 → CW = back.
                # We skip BACK-facing (winding > 0) when in solid/non-wireframe
                # mode.  Allow degenerate tris (winding ≈ 0) through to avoid
                # holes along silhouette edges.
                ex1 = p1[0] - p0[0]; ey1 = p1[1] - p0[1]
                ex2 = p2[0] - p0[0]; ey2 = p2[1] - p0[1]
                winding = ex1 * ey2 - ex2 * ey1
                # Skip back-facing (CW in screen-Y-down space)
                if winding > 0 and not self.show_wireframe and self.show_solid and not is_two_sided_flat:
                    continue

                # Use weighted-centroid depth: average is more stable than
                # min() for coplanar/nearly-coplanar faces (fixes bantha Z-fighting).
                # Small face-index jitter breaks ties deterministically.
                depth = (p0[2] + p1[2] + p2[2]) * 0.3333 + fi * 1e-7

                # Normal — use world-space normals for correct lighting
                if n_norms > max(v0, v1, v2):
                    nx = (world_norms[v0][0]+world_norms[v1][0]+world_norms[v2][0]) / 3.0
                    ny = (world_norms[v0][1]+world_norms[v1][1]+world_norms[v2][1]) / 3.0
                    nz = (world_norms[v0][2]+world_norms[v1][2]+world_norms[v2][2]) / 3.0
                    nl_len = math.sqrt(nx*nx+ny*ny+nz*nz)
                    if nl_len > 1e-9:
                        nx /= nl_len; ny /= nl_len; nz /= nl_len
                    norm = (nx, ny, nz)
                else:
                    norm = self._face_normal(wv0, wv1, wv2)

                lx, ly, lz = self._light_dir
                ndotl = _clamp(_dot(norm, (lx, ly, lz)), 0.0, 1.0)
                ndotl = max(ndotl, _clamp(-_dot(norm, (lx, ly, lz)), 0.0, 1.0) * (0.55 if is_two_sided_flat else 0.35))
                shade = self._ambient + (1.0 - self._ambient) * ndotl
                fill  = (int(r*shade), int(g*shade), int(b*shade))

                # Depth bias for transparent tris to sort after opaque at same depth
                sort_depth = depth - (1e-3 if node_alpha < 0.999 else 0.0)
                # UE-inspired: convert to sortable uint key for stable integer comparison
                sort_key = _float_to_sort_key(sort_depth)
                # Two-pass tier: opaque=0, transparent/additive=1.
                # Tier is the PRIMARY sort dimension — all opaque tris are drawn
                # before any transparent tri regardless of depth.  This prevents
                # transparent inner geometry (eyes, teeth) from rendering on top
                # of the opaque face mesh purely because of centroid-depth ordering.
                _th_flat = int(getattr(node, 'transparency_hint', 0))
                # Inner-geometry (eyes, eyelids, teeth) are promoted to tier 1
                # even when transparency_hint==0 so they draw AFTER the opaque
                # face/head mesh.  This exposes them through the eye-socket and
                # mouth-gap openings in the face geometry.
                _is_trans_flat = (_th_flat > 0 or node_alpha < 0.999 or _is_inner_geo_flat)
                # Background geometry must be a distinct first pass.  A tiny
                # centroid-depth bias is not sufficient for giant sky panels:
                # their centroid can sort in front of ordinary room triangles
                # and paint an opaque sky over the map in this CPU painter.
                _tier_flat = -1 if bool(getattr(node, 'background_geometry', False)) else (1 if _is_trans_flat else 0)
                tris.append((sort_key, ((p0[0],p0[1]), (p1[0],p1[1]), (p2[0],p2[1])), fill, is_sel, fi, node_alpha, _tier_flat))
                if len(tris) >= tri_cap:
                    break
            if len(tris) >= tri_cap:
                break

        # Two-pass sort: tier 0 (opaque) before tier 1 (transparent);
        # within each tier, back-to-front by depth; ties broken by face index.
        # This prevents transparent inner geometry (eyes, hair, teeth, gums)
        # from rendering on top of opaque face/body meshes when centroid depth
        # ordering alone would place them in front.
        tris.sort(key=lambda t: (t[6], -t[0], t[4]))

        for depth, pts, fill, is_sel, _fi, t_alpha, _tier in tris:
            flat = [pts[0][0],pts[0][1], pts[1][0],pts[1][1], pts[2][0],pts[2][1]]
            if self.show_solid:
                sel_fill = (min(fill[0]+30,255), min(fill[1]+50,255), fill[2]) if is_sel else fill
                if t_alpha < 0.999:
                    # Blend with background colour for transparent flat-shaded faces
                    bg = tuple(getattr(self, "viewport_background", _BG[:3]))
                    a = t_alpha
                    sel_fill = (int(sel_fill[0]*a + bg[0]*(1-a)),
                                int(sel_fill[1]*a + bg[1]*(1-a)),
                                int(sel_fill[2]*a + bg[2]*(1-a)))
                draw.polygon(flat, fill=sel_fill)
            if self.show_wireframe or is_sel:
                wire_col = _SEL[:3] if is_sel else _WIRE[:3]
                draw.polygon(flat, outline=wire_col)

        # NOTE: _draw_bones is called by render() with a fresh draw context.

    # ── Accelerated rasterizer (v10.5) ────────────────────────────────
    # Uses accel.py (Numba JIT tier 1 or NumPy tier 2) for 17–40× speedup
    # over the PIL AFFINE path.  Falls back to PIL if accel is unavailable.

    def _draw_mesh_accel(self, draw: 'ImageDraw.Draw',
                         img: 'Image.Image', W: int, H: int,
                         flat_only: bool = False) -> bool:
        """
        Batch rasterizer using the accel.py acceleration layer.

        When flat_only=True (interactive drag), uses flat_shade_frame_jit for
        maximum speed (~100 fps on high-poly models).
        When flat_only=False (textured idle), uses rasterize_frame_jit with
        per-triangle UV sampling.

        Returns True if the accel path ran, False if it should fall back to PIL.

        Architecture (v10.5):
        1. Collect world verts + UVs per node (same as _draw_mesh_textured).
        2. _proj_batch → NumPy vectorized screen projection.
        3. frustum_cull_np → vectorized AABB cull.
        4. depth_sort_np → NumPy argsort (3× faster than Python sort).
        5. _accel_rasterize_frame / _accel_flat_shade_frame → batch rasterize.
        6. Convert NumPy framebuffer back to PIL for compositing.
        """
        if not _ACCEL_AVAILABLE or not _NUMPY:
            return False
        if not _PIL:
            return False

        cam       = self.cam
        light_dir = self._light_dir
        ambient   = self._ambient

        # Use accel cap (10k) for textured; interactive is also 10k (same limit)
        tri_cap = min(self._screen_size_lod_cap(W, H),
                      self.MAX_TRIS_TEXTURED_ACCEL if not flat_only
                      else self.MAX_TRIS_INTERACTIVE)

        # ── 1. Allocate NumPy framebuffer ─────────────────────────────────
        # We maintain a separate NumPy (H, W, 4) RGBA buffer so the JIT
        # rasterizer can write pixels directly without PIL overhead.
        # Pre-fill with the viewport background colour.
        bg_r, bg_g, bg_b = tuple(getattr(self, "viewport_background", _BG[:3]))
        buf = np.empty((H, W, 4), dtype=np.uint8)
        buf[:, :, 0] = bg_r
        buf[:, :, 1] = bg_g
        buf[:, :, 2] = bg_b
        buf[:, :, 3] = 255

        # Copy existing img pixels (e.g. grid) into buf so grid is preserved
        try:
            existing = np.array(img, dtype=np.uint8)
            if existing.shape == (H, W, 4):
                buf[:] = existing
            elif existing.shape == (H, W, 3):
                buf[:, :, :3] = existing
                buf[:, :, 3] = 255
        except Exception:
            pass  # If img copy fails, use plain bg (grid will be redrawn after)

        # ── 2. Collect all visible triangles ─────────────────────────────
        # Per-node arrays are built then concatenated at the end for the batch call.
        # For multi-texture models we make one rasterize_frame call per texture batch.
        # For simplicity in v10.5 we store: [(tex_arr, verts_sx, verts_sy, uvs_u,
        #   uvs_v, fv0, fv1, fv2, depths, shade_r, shade_g, shade_b, alphas), ...]
        # One entry per (node, texture) pair.
        batches = []   # list of dicts, one per unique (node, texture)

        total_tris = 0
        wire_tris  = []  # [(flat_pts, wire_col), ...]

        for node in self._iter_visible_mesh_nodes():
            if not node.vertices or not node.faces:
                continue
            if total_tris >= tri_cap:
                break

            verts   = node.vertices
            nv      = len(verts)
            uvs     = node.uvs if not flat_only else []
            n_uvs   = len(uvs)
            face_uvs_list = getattr(node, 'face_uvs', [])
            _has_face_uvs = bool(face_uvs_list) and len(face_uvs_list) == len(node.faces)
            has_uvs = (n_uvs > 0) and not flat_only
            is_sel  = (node is self.selected_node)

            # ── World transform + projection ──────────────────────────────
            _node_id = id(node)
            _fvc = getattr(self, '_frame_verts_cache', None)
            _fnc = getattr(self, '_frame_norms_cache', None)
            if _fvc is not None and _node_id in _fvc:
                world_verts = _fvc[_node_id]
            else:
                world_verts = self._get_world_verts_for_node(node)
                if _fvc is not None:
                    _fvc[_node_id] = world_verts

            if _fnc is not None and _node_id in _fnc:
                world_norms = _fnc[_node_id]
            else:
                world_norms = self._get_world_normals_for_node(node)
                if _fnc is not None:
                    _fnc[_node_id] = world_norms
            n_norms = len(world_norms)

            if n_norms == 0 and world_verts and node.faces:
                world_norms = self._compute_area_weighted_normals(node.faces, world_verts)
                n_norms = len(world_norms)

            # ── Batch project all vertices via NumPy ─────────────────────
            screen_verts_t = self._proj_batch(world_verts, W, H)
            # Build sx/sy/valid arrays
            sv_sx = np.full(nv, -9999, dtype=np.int32)
            sv_sy = np.full(nv, -9999, dtype=np.int32)
            sv_cz = np.zeros(nv, dtype=np.float32)
            sv_ok = np.zeros(nv, dtype=np.bool_)
            for i, p in enumerate(screen_verts_t):
                if p is not None:
                    sv_sx[i], sv_sy[i], sv_cz[i] = p
                    sv_ok[i] = True

            # ── Per-node texture & diffuse colour ─────────────────────────
            _use_lq = self._lq_tex_mode
            # FIX-LMROUTE: When has_lightmap=True, tex_count==2 means
            # slot 0 = diffuse and slot 1 = lightmap.  The lightmap is
            # composited as a separate multiply pass (not per-face material).
            # Treating lightmapped nodes as multi-texture causes face_mats[i]=1
            # to route ALL faces to the lightmap image as their diffuse texture,
            # which is the "texture-to-face routing" bug (D5).  xoreos and
            # KotOR.js both handle textureIndex==1 as lightmap, not per-face
            # material selection.
            _node_has_lightmap_accel = bool(getattr(node, 'has_lightmap', False))
            _node_is_multitex = (getattr(node, 'tex_count', 1) > 1
                                 and bool(getattr(node, 'face_mats', []))
                                 and bool(getattr(node, 'texture_names', []))
                                 and not _node_has_lightmap_accel)
            node_alpha = float(_clamp(getattr(node, 'alpha', 1.0), 0.0, 1.0))
            # transparency_hint is a render-mode flag, not an alpha override.
            # Only explicit node.alpha < 1 or CTRL_MESH_ALPHA animation sets transparency.

            # Animation overrides
            if self._anim_pose is not None:
                _pn = _pose_node_for_transform(node, self._anim_pose)
                if _pn is not None and _pn.alpha is not None:
                    node_alpha = _clamp(_pn.alpha, 0.0, 1.0)

            clean_tex = _clean_tex_name(node.texture)
            if not clean_tex or clean_tex.upper() in ('NULL', ''):
                diff = (0.55, 0.55, 0.65)
            else:
                diff = (
                    _clamp(node.diffuse[0], 0.0, 1.0),
                    _clamp(node.diffuse[1], 0.0, 1.0),
                    _clamp(node.diffuse[2], 0.0, 1.0),
                )

            selfillum = getattr(node, 'selfillum', (0.0, 0.0, 0.0))
            if self._anim_pose is not None:
                _pn_si = _pose_node_for_transform(node, self._anim_pose)
                if _pn_si is not None and _pn_si.selfillum is not None:
                    selfillum = _pn_si.selfillum

            # Get single-tex image + array once per node
            if not flat_only and not _node_is_multitex and has_uvs:
                _raw_tex = self._get_tex(node)
                _pil_tex = self.tex_cache.get_mip1(_raw_tex) if (_use_lq and _raw_tex) else _raw_tex
                _tex_arr = self._tex_arr_cache.get(_pil_tex) if _pil_tex else None
            else:
                _pil_tex = None
                _tex_arr = None

            transp_hint = getattr(node, 'transparency_hint', 0)
            _nl_accel = node.name.lower()
            _is_face_accel = any(s in _nl_accel for s in _FACE_MESH_SUBSTRINGS)
            # BUG FIX v26: inner-geo nodes two-sided in accel path too
            _is_inner_geo_accel = any(s in _nl_accel for s in _INNER_GEO_SUBSTRINGS)
            is_two_sided = (node.is_dangly
                            or transp_hint in (1, 2)
                            or _is_face_accel
                            or _is_inner_geo_accel)

            # ── Per-node TXI features (Phase 18-C) ────────────────────────
            # TXI clamp_s/clamp_t: apply GL_CLAMP_TO_EDGE on the relevant axis.
            # This makes the accel path match the PIL path for clamped textures.
            _accel_clamp_s = bool(getattr(node, 'txi_clamp_s', False))
            _accel_clamp_t = bool(getattr(node, 'txi_clamp_t', False))
            # FIX-EDGEBLEED (GPU): Keep the accelerated path in lockstep with the
            # CPU/PIL path for single-tile atlas meshes.  Custom override MDLs often
            # use normal 0..1 character atlases without TXI clamp flags; wrapping
            # those UVs makes armor panels sample the opposite edge of the atlas.
            # Nodes with true tiled UVs or animated/procedural TXI still use repeat.
            # Shared single-tile-atlas test (see gpu_diagnostics_records):
            # clamps character/item atlases whose UV island stays within one
            # tile even if a few verts overshoot [0,1], instead of the old
            # first-30-vertex sample that both missed later overshoots and
            # depended on vertex ordering.
            if (not _accel_clamp_s or not _accel_clamp_t) and _node_uses_single_tile_atlas(node):
                _accel_clamp_s = True
                _accel_clamp_t = True
            # UV animation (animate_uv): add time-based scroll offset.
            _accel_animate_uv = bool(getattr(node, 'animate_uv', False))
            _accel_uv_scroll_u = 0.0
            _accel_uv_scroll_v = 0.0
            if _accel_animate_uv:
                _accel_uv_dir_x = float(getattr(node, 'uv_dir_x', 0.0) or 0.0)
                _accel_uv_dir_y = float(getattr(node, 'uv_dir_y', 0.0) or 0.0)
                _accel_uv_jitter = float(getattr(node, 'uv_jitter', 0.0) or 0.0)
                _accel_uv_jitter_spd = float(getattr(node, 'uv_jitter_speed', 0.0) or 0.0)
                _t_anim = getattr(self, '_anim_time', 0.0)
                if _accel_uv_dir_x != 0.0 or _accel_uv_dir_y != 0.0:
                    _accel_uv_scroll_u = _accel_uv_dir_x * _t_anim
                    _accel_uv_scroll_v = _accel_uv_dir_y * _t_anim
                if _accel_uv_jitter != 0.0 and _accel_uv_jitter_spd > 0.0:
                    import random as _random
                    _jitter = _random.uniform(-_accel_uv_jitter, _accel_uv_jitter)
                    _accel_uv_scroll_u += _jitter
                    _accel_uv_scroll_v += _jitter
            # rotatetexture: rotate UV 90° CCW = (u, v) → (v, 1-u)
            _accel_rotate_tex = bool(getattr(node, 'rotatetexture', False)
                                     or getattr(node, 'rotate_texture', False))

            # ── Per-face loop ─────────────────────────────────────────────
            # Build per-face arrays for this node's triangles
            face_x0 = []; face_y0 = []; face_x1 = []; face_y1 = []
            face_x2 = []; face_y2 = []
            face_u0 = []; face_v0_l = []; face_u1 = []; face_v1_l = []
            face_u2 = []; face_v2_l = []
            face_sr = []; face_sg = []; face_sb = []
            face_alpha = []
            face_depths = []
            face_is_sel = []
            face_tex_arr = []   # per-face tex array (for multi-tex)
            fi_insert = []      # insertion order for Z-tie breaking

            for _fi, face in enumerate(node.faces):
                if len(face) < 3:
                    continue
                vi0, vi1, vi2 = face[0], face[1], face[2]
                if vi0 == vi1 or vi1 == vi2 or vi0 == vi2:
                    continue
                if vi0 >= nv or vi1 >= nv or vi2 >= nv:
                    continue
                if not (sv_ok[vi0] and sv_ok[vi1] and sv_ok[vi2]):
                    continue

                p0 = (sv_sx[vi0], sv_sy[vi0], sv_cz[vi0])
                p1 = (sv_sx[vi1], sv_sy[vi1], sv_cz[vi1])
                p2 = (sv_sx[vi2], sv_sy[vi2], sv_cz[vi2])

                # Backface cull
                ex1 = float(p1[0])-float(p0[0]); ey1 = float(p1[1])-float(p0[1])
                ex2 = float(p2[0])-float(p0[0]); ey2 = float(p2[1])-float(p0[1])
                winding = ex1*ey2 - ex2*ey1
                if winding > 0 and not self.show_wireframe and self.show_solid and not is_two_sided:
                    continue

                fi_local = len(face_depths)
                depth = (p0[2] + p1[2] + p2[2]) * 0.3333 + fi_local * 1e-7

                # ── UV resolve ────────────────────────────────────────────
                if has_uvs and not flat_only:
                    if _has_face_uvs:
                        fuv = face_uvs_list[_fi]
                        ti0, ti1, ti2 = fuv[0], fuv[1], fuv[2]
                    else:
                        ti0, ti1, ti2 = vi0, vi1, vi2
                    uv0 = uvs[ti0] if ti0 < n_uvs else (0.5, 0.5)
                    uv1 = uvs[ti1] if ti1 < n_uvs else (0.5, 0.5)
                    uv2 = uvs[ti2] if ti2 < n_uvs else (0.5, 0.5)

                    # GL_REPEAT handles legitimate tiled UV coordinates.
                    _uv_sum = (uv0[0] + uv0[1] + uv1[0] + uv1[1] + uv2[0] + uv2[1])
                    if _uv_sum != _uv_sum:  # NaN check (NaN != NaN)
                        continue

                    u0r, u1r, u2r = uv0[0], uv1[0], uv2[0]
                    v0r, v1r, v2r = uv0[1], uv1[1], uv2[1]

                    # ── Phase 18-C: TXI clamp (GL_CLAMP_TO_EDGE) ─────────────
                    # Clamp UVs to [0, 1-eps] on axes that have TXI clamp set.
                    # The upper bound is 1-eps (not 1.0) because the accel rasterizer
                    # applies frac() per pixel: frac(1.0)=0.0 would sample the wrong
                    # edge. GL_CLAMP_TO_EDGE should sample the LAST texel, so we
                    # clamp to TW-1/TW ≈ 0.9990... Using a small epsilon is correct.
                    # This prevents tiling on head textures, decals, etc.
                    # Also skip the seam fix on clamped axes (no tiling = no seam).
                    _CLAMP_MAX = 0.9999  # just below 1.0 so frac() stays near edge
                    if _accel_clamp_s:
                        u0r = max(0.0, min(_CLAMP_MAX, u0r))
                        u1r = max(0.0, min(_CLAMP_MAX, u1r))
                        u2r = max(0.0, min(_CLAMP_MAX, u2r))
                    if _accel_clamp_t:
                        v0r = max(0.0, min(_CLAMP_MAX, v0r))
                        v1r = max(0.0, min(_CLAMP_MAX, v1r))
                        v2r = max(0.0, min(_CLAMP_MAX, v2r))

                    # ── Phase 18-D: rotatetexture (90° CCW UV rotation) ───────
                    # KotOR rotatetexture: (u, v) → (v, 1-u)
                    if _accel_rotate_tex:
                        u0r, v0r = v0r, 1.0 - u0r
                        u1r, v1r = v1r, 1.0 - u1r
                        u2r, v2r = v2r, 1.0 - u2r

                    # ── Phase 18-D: UV animation (animate_uv scroll) ──────────
                    # Add time-based scroll offset. The accel rasterizer's frac()
                    # handles modulo wrap automatically, so no clamping needed here.
                    if _accel_uv_scroll_u != 0.0:
                        u0r += _accel_uv_scroll_u
                        u1r += _accel_uv_scroll_u
                        u2r += _accel_uv_scroll_u
                    if _accel_uv_scroll_v != 0.0:
                        v0r += _accel_uv_scroll_v
                        v1r += _accel_uv_scroll_v
                        v2r += _accel_uv_scroll_v

                    # Resolve the face image before the V-orientation step. A
                    # loose DCC-authored TGA can opt out of the renderer's
                    # KotOR/D3D V conversion while stock textures retain it.
                    if _node_is_multitex:
                        _raw_ft = self._get_tex_for_face(node, _fi)
                        _pil_ft = self.tex_cache.get_mip1(_raw_ft) if (_use_lq and _raw_ft) else _raw_ft
                        _face_pil_tex = _pil_ft
                        _ta = self._tex_arr_cache.get(_pil_ft) if _pil_ft else None
                    else:
                        _face_pil_tex = _pil_tex
                        _ta = _tex_arr
                    _effective_v_flip = (
                        bool(getattr(node, "uv_v_flip", True))
                        and bool(getattr(_face_pil_tex, "_gr_gpu_uv_v_flip", True))
                    )
                    if not _effective_v_flip:
                        # The accelerated rasterizer applies 1-v internally;
                        # pre-flip to cancel it for bottom-left DCC UVs.
                        v0r = 1.0 - v0r
                        v1r = 1.0 - v1r
                        v2r = 1.0 - v2r

                    # Seam fix (reuse existing helpers)
                    # Only apply when span < 1.0 — multi-tile faces (span >= 1.0)
                    # are handled by the accel rasterizer's frac() UV wrapping and
                    # must NOT be seam-fixed (would collapse tile range to zero span).
                    # Also skip on clamped axes (clamp + seam fix would interfere).
                    raw_span_u = max(u0r, u1r, u2r) - min(u0r, u1r, u2r)
                    raw_span_v = max(v0r, v1r, v2r) - min(v0r, v1r, v2r)
                    if raw_span_u < 1.0 and not _accel_clamp_s:
                        u_has_seam = (_edge_has_seam_global(u0r, u1r) or
                                      _edge_has_seam_global(u0r, u2r) or
                                      _edge_has_seam_global(u1r, u2r))
                        if u_has_seam:
                            u1r = _uwrap_global(u0r, u1r)
                            u2r = _uwrap_global(u0r, u2r)
                    if raw_span_v < 1.0 and not _accel_clamp_t:
                        v_has_seam = (_edge_has_seam_global(v0r, v1r) or
                                      _edge_has_seam_global(v0r, v2r) or
                                      _edge_has_seam_global(v1r, v2r))
                        if v_has_seam:
                            v1r = _uwrap_global(v0r, v1r)
                            v2r = _uwrap_global(v0r, v2r)
                    uv0 = (u0r, v0r); uv1 = (u1r, v1r); uv2 = (u2r, v2r)

                    face_tex_arr.append(_ta)
                else:
                    uv0 = uv1 = uv2 = (0.5, 0.5)
                    face_tex_arr.append(None)

                # ── Per-face lighting ─────────────────────────────────────
                if n_norms > max(vi0, vi1, vi2):
                    nx = (world_norms[vi0][0] + world_norms[vi1][0] + world_norms[vi2][0]) / 3.0
                    ny = (world_norms[vi0][1] + world_norms[vi1][1] + world_norms[vi2][1]) / 3.0
                    nz = (world_norms[vi0][2] + world_norms[vi1][2] + world_norms[vi2][2]) / 3.0
                    nl = math.sqrt(nx*nx + ny*ny + nz*nz)
                    if nl > 1e-9:
                        nx /= nl; ny /= nl; nz /= nl
                else:
                    wv0 = world_verts[vi0]; wv1 = world_verts[vi1]; wv2 = world_verts[vi2]
                    fnorm = self._face_normal(wv0, wv1, wv2)
                    nx, ny, nz = fnorm

                ndotl = nx*light_dir[0] + ny*light_dir[1] + nz*light_dir[2]
                ndotl_f = max(0.0, ndotl) + max(0.0, -ndotl) * (0.55 if is_two_sided else 0.35)
                shade = ambient + (1.0 - ambient) * ndotl_f

                dr, dg, db = diff
                si_r, si_g, si_b = selfillum
                shade_r = int(_clamp((shade + si_r) * (0.5 + dr*0.5) * 255, 0, 255))
                shade_g = int(_clamp((shade + si_g) * (0.5 + dg*0.5) * 255, 0, 255))
                shade_b = int(_clamp((shade + si_b) * (0.5 + db*0.5) * 255, 0, 255))

                face_x0.append(int(p0[0])); face_y0.append(int(p0[1]))
                face_x1.append(int(p1[0])); face_y1.append(int(p1[1]))
                face_x2.append(int(p2[0])); face_y2.append(int(p2[1]))
                face_u0.append(uv0[0]); face_v0_l.append(uv0[1])
                face_u1.append(uv1[0]); face_v1_l.append(uv1[1])
                face_u2.append(uv2[0]); face_v2_l.append(uv2[1])
                face_sr.append(shade_r); face_sg.append(shade_g); face_sb.append(shade_b)
                face_alpha.append(node_alpha)
                face_depths.append(depth)
                face_is_sel.append(is_sel)
                fi_insert.append(fi_local)

                total_tris += 1
                if total_tris >= tri_cap:
                    break
            if not face_depths:
                if total_tris >= tri_cap:
                    break
                continue

            # ── Sort and rasterize this node's batch ──────────────────────
            NF = len(face_depths)
            depths_arr = np.array(face_depths, dtype=np.float32)
            order = _accel_depth_sort(depths_arr)  # back-to-front indices

            # Build sorted arrays
            sx0 = np.array(face_x0, dtype=np.int64)[order]
            sy0 = np.array(face_y0, dtype=np.int64)[order]
            sx1 = np.array(face_x1, dtype=np.int64)[order]
            sy1 = np.array(face_y1, dtype=np.int64)[order]
            sx2 = np.array(face_x2, dtype=np.int64)[order]
            sy2 = np.array(face_y2, dtype=np.int64)[order]

            # Frustum cull
            sc_x = np.stack([sx0, sx1, sx2], axis=1)
            sc_y = np.stack([sy0, sy1, sy2], axis=1)
            visible = _accel_frustum_cull(sc_x, sc_y, W, H)

            sr_arr = np.array(face_sr, dtype=np.int64)[order]
            sg_arr = np.array(face_sg, dtype=np.int64)[order]
            sb_arr = np.array(face_sb, dtype=np.int64)[order]
            alpha_arr = np.array(face_alpha, dtype=np.float64)[order]

            # Determine whether to use textured or flat pass.
            # Use textured pass when:
            #   1. NOT in flat_only (interactive drag) mode, AND
            #   2. Either the node-level _tex_arr is set (single-tex fast path),
            #      OR at least one face in face_tex_arr has a texture (multi-tex
            #      OR case where the cache was just populated during prewarm).
            # Previously this condition was `flat_only or _tex_arr is None` which
            # meant multi-texture nodes ALWAYS rendered flat because _tex_arr is
            # intentionally None for those nodes (line ~5088).  It also meant
            # single-tex nodes fell to flat if the texture array hadn't been
            # converted to NumPy yet (TexArrayCache miss on first frame).
            _any_tex_arr = (_tex_arr is not None or
                            any(t is not None for t in face_tex_arr))
            if flat_only or not _any_tex_arr:
                # ── Flat shade pass ────────────────────────────────────────
                # Build synthetic per-vertex arrays for single flat triangle
                # rasterization: vertex 0 = p0, vertex 1 = p1, vertex 2 = p2,
                # face references vertex indices 0,1,2 directly.
                # We abuse the flat_shade_frame API which expects global vertex arrays.
                # Build (3*NF,) vertex arrays with one triangle per 3 vertices.
                _nx = NF
                all_sx = np.empty(_nx * 3, dtype=np.int64)
                all_sy = np.empty(_nx * 3, dtype=np.int64)
                all_sx[0::3] = sx0; all_sy[0::3] = sy0
                all_sx[1::3] = sx1; all_sy[1::3] = sy1
                all_sx[2::3] = sx2; all_sy[2::3] = sy2
                fv0 = np.arange(0, _nx*3, 3, dtype=np.int64)
                fv1 = fv0 + 1
                fv2 = fv0 + 2
                fr_arr = np.clip(sr_arr, 0, 255).astype(np.uint8)
                fg_arr = np.clip(sg_arr, 0, 255).astype(np.uint8)
                fb_arr = np.clip(sb_arr, 0, 255).astype(np.uint8)
                _accel_flat_shade_frame(buf, all_sx, all_sy, fv0, fv1, fv2,
                                        fr_arr, fg_arr, fb_arr, visible)
            else:
                # ── Textured pass ──────────────────────────────────────────
                # Group sorted faces by their texture array.
                # In the common single-texture case this is one group.
                # Faces with None texture fall back to flat-shade within
                # this same pass (avoids a separate flat-shade call for models
                # that have a mix of textured and untextured faces).
                ordered_tex = [face_tex_arr[i] for i in order]
                u0a = np.array(face_u0, dtype=np.float64)[order]
                v0a = np.array(face_v0_l, dtype=np.float64)[order]
                u1a = np.array(face_u1, dtype=np.float64)[order]
                v1a = np.array(face_v1_l, dtype=np.float64)[order]
                u2a = np.array(face_u2, dtype=np.float64)[order]
                v2a = np.array(face_v2_l, dtype=np.float64)[order]

                # Build 3*NF vertex arrays for the batch call
                _nx = NF
                all_sx = np.empty(_nx * 3, dtype=np.int64)
                all_sy = np.empty(_nx * 3, dtype=np.int64)
                all_sx[0::3] = sx0; all_sy[0::3] = sy0
                all_sx[1::3] = sx1; all_sy[1::3] = sy1
                all_sx[2::3] = sx2; all_sy[2::3] = sy2
                # Per-vertex UV (one per triangle-vertex)
                all_uu = np.empty(_nx * 3, dtype=np.float64)
                all_vv = np.empty(_nx * 3, dtype=np.float64)
                all_uu[0::3] = u0a; all_uu[1::3] = u1a; all_uu[2::3] = u2a
                all_vv[0::3] = v0a; all_vv[1::3] = v1a; all_vv[2::3] = v2a
                fv0 = np.arange(0, _nx*3, 3, dtype=np.int64)
                fv1 = fv0 + 1
                fv2 = fv0 + 2

                # Group by texture for batch calls.
                # Most nodes are single-texture → one call.
                # Faces with None texture get a flat-shade call instead.
                prev_tex = None
                group_start = 0
                _flat_vis_list = []   # indices of visible None-tex faces for flat fallback
                for gi in range(NF + 1):
                    cur_tex = ordered_tex[gi] if gi < NF else None
                    if cur_tex is not prev_tex or gi == NF:
                        # Flush previous group
                        if gi > group_start:
                            g_sl = slice(group_start, gi)
                            g_fv0 = fv0[g_sl]; g_fv1 = fv1[g_sl]; g_fv2 = fv2[g_sl]
                            if prev_tex is not None:
                                _accel_rasterize_frame(
                                    buf, prev_tex,
                                    all_sx, all_sy,
                                    all_uu, all_vv,
                                    g_fv0, g_fv1, g_fv2,
                                    sr_arr[g_sl], sg_arr[g_sl], sb_arr[g_sl],
                                    alpha_arr[g_sl],
                                    visible[g_sl],
                                    clamp_s=_accel_clamp_s,
                                    clamp_t=_accel_clamp_t,
                                )
                            else:
                                # No texture for this group — render as flat-shade
                                fr_g = np.clip(sr_arr[g_sl], 0, 255).astype(np.uint8)
                                fg_g = np.clip(sg_arr[g_sl], 0, 255).astype(np.uint8)
                                fb_g = np.clip(sb_arr[g_sl], 0, 255).astype(np.uint8)
                                _accel_flat_shade_frame(
                                    buf, all_sx, all_sy,
                                    g_fv0, g_fv1, g_fv2,
                                    fr_g, fg_g, fb_g,
                                    visible[g_sl],
                                )
                        prev_tex   = cur_tex
                        group_start = gi

            # Collect wireframe data
            if self.show_wireframe or is_sel:
                for idx_sorted in range(NF):
                    if not visible[idx_sorted]:
                        continue
                    flat = [int(sx0[idx_sorted]), int(sy0[idx_sorted]),
                            int(sx1[idx_sorted]), int(sy1[idx_sorted]),
                            int(sx2[idx_sorted]), int(sy2[idx_sorted])]
                    orig_is_sel = face_is_sel[order[idx_sorted]]
                    wire_col = _SEL[:3] if orig_is_sel else _WIRE[:3]
                    wire_tris.append((flat, wire_col))

            if total_tris >= tri_cap:
                break

        # ── 3. Convert NumPy buffer back to PIL ───────────────────────────
        try:
            result_img = Image.fromarray(buf, 'RGBA')
            # Blit result into img in-place
            img.paste(result_img, (0, 0))
        except Exception as exc:
            log.debug(f"_draw_mesh_accel: PIL conversion failed ({exc})")
            return False

        # ── 4. Wireframe pass ─────────────────────────────────────────────
        if wire_tris:
            draw2 = ImageDraw.Draw(img)
            for flat, wire_col in wire_tris:
                draw2.polygon(flat, outline=wire_col)

        return True  # accel path ran successfully

    # ── Full UV-mapped textured renderer (fast PIL-based) ──────────────

    def _draw_mesh_textured(self, draw: 'ImageDraw.Draw',
                             img: 'Image.Image', W: int, H: int):
        """
        UV-mapped textured rendering using PIL AFFINE transform per triangle.

        Strategy:
        1. Collect ALL triangles from visible nodes with world-space vertices,
           UVs, normals and depth → sort back-to-front (painter's algorithm)
        2. For each triangle with a loaded texture:
           a. Compute per-face lighting (center normal dot light)
           b. Use _paste_textured_triangle() with PIL AFFINE transform for
              proper per-pixel UV interpolation (no pixelation from centroid sampling)
           c. Modulate with per-face lighting shade color
        3. For triangles WITHOUT a loaded texture: flat-fill with diffuse color

        The AFFINE warp correctly maps the texture onto each triangle using bilinear
        sampling, eliminating the pixelation caused by the old centroid-only approach.
        """
        cam = self.cam
        light_dir = self._light_dir
        ambient   = self._ambient

        # Use reduced tri cap during interactive drag for fast viewport response
        # For textured mode use a smaller cap (PIL affine per-tri is slow)
        tri_cap = min(self._screen_size_lod_cap(W, H), self.MAX_TRIS_TEXTURED)

        # ── Collect all triangles ────────────────────────────────────────
        # Entry: (depth, screen_pts, fill_rgb, tex_img, uv0, uv1, uv2, is_sel)
        tris = []

        for node in self._iter_visible_mesh_nodes():
            if not node.vertices or not node.faces:
                continue

            verts   = node.vertices
            nv      = len(verts)
            uvs     = node.uvs
            n_uvs   = len(uvs)
            # face_uvs: per-face tvert index triples (ASCII MDL only).
            # When present, uvs[face_uvs[fi][k]] gives the UV for face fi, vertex k.
            # When absent (binary MDL), use vertex indices directly.
            face_uvs_list = getattr(node, 'face_uvs', [])
            _has_face_uvs = bool(face_uvs_list) and len(face_uvs_list) == len(node.faces)
            n_norms = 0        # will be set after world_norms computed
            has_uvs = (n_uvs > 0)
            is_sel  = (node is self.selected_node)

            # Multi-texture support: does this node use per-face texture selection?
            # tex_count > 1 means face_mats[i] indexes into texture_names[slot].
            # Single-texture fast-path: pre-resolve once per node.
            _node_tex_count = getattr(node, 'tex_count', 1)
            # FIX-LMROUTE-V2: Determine lightmap status BEFORE multitex check.
            # _node_has_lm must be computed here (not later at lightmap-setup)
            # because _node_is_multitex depends on it.  The original code defined
            # _node_has_lm 17 lines AFTER its first use, causing a NameError on
            # the first loop iteration and stale-value bugs on subsequent nodes.
            # This was the root cause of the D5 texture-to-face routing bug in
            # the PIL AFFINE fallback path (_draw_mesh_textured).
            _node_has_lm = bool(getattr(node, 'has_lightmap', False))
            # FIX-LMROUTE: Lightmapped nodes (has_lightmap=True) must NOT
            # be treated as multi-texture even when tex_count==2.  In KotOR,
            # slot 1 is the lightmap (composited via UV1 multiply pass), not a
            # per-face material variant.  face_mats[i]==1 on lightmapped nodes
            # means "this face has a lightmap", NOT "use texture_names[1] as
            # diffuse".  Without this guard, _get_tex_for_face routes all faces
            # to the lightmap image as their primary diffuse texture, producing
            # the D5 "texture-to-face routing" bug.
            # Reference: xoreos setupShaderTexture (textureIndex==1 → LIGHTMAP,
            #            not per-face material); KotOR.js textureMap2 = lightmap.
            _node_is_multitex = (_node_tex_count > 1
                                 and bool(getattr(node, 'face_mats', []))
                                 and bool(getattr(node, 'texture_names', []))
                                 and not _node_has_lm)
            # For single-texture nodes resolve once; multi-tex resolves per face.
            # PERF-FIX (v10.2): When _lq_tex_mode is active (first frame after drag
            # release), use mip1 (half-res) textures to halve the PIL AFFINE warp
            # cost.  TextureCache.get_mip1() is O(1) after the first access.
            _use_lq = self._lq_tex_mode  # FIX (v10.4): now always a real attr
            if not _node_is_multitex and has_uvs:
                _raw_tex = self._get_tex(node)
                tex_img = self.tex_cache.get_mip1(_raw_tex) if (_use_lq and _raw_tex is not None) else _raw_tex
            else:
                tex_img = None

            # ── Lightmap setup ─────────────────────────────────────────────
            # KotOR lightmaps: has_lightmap=True means node.lightmap holds the
            # lightmap texture name; node.uvs_lm holds per-vertex lightmap UVs.
            # We load the lightmap image once per node and composite it as a
            # multiply pass (overbright ×2) after the diffuse pass.
            # NOTE: _node_has_lm was already computed above (FIX-LMROUTE-V2)
            # for the multitex check.  No need to re-compute here.
            _lm_tex_name   = str(getattr(node, 'lightmap', ''))
            _lm_override_path = str(getattr(node, '_gr_baked_lightmap_preview_path', '') or getattr(node, '_gr_baked_lightmap_path', '') or '')
            _lm_override_name = str(getattr(node, '_gr_baked_lightmap_preview_name', '') or '')
            _uvs_lm        = getattr(node, 'uvs_lm', [])
            _n_uvs_lm      = len(_uvs_lm)
            _has_lm_uvs    = (_n_uvs_lm > 0)
            lm_img = None
            if _lm_override_path and _has_lm_uvs:
                lm_img = self._get_image_by_path(_lm_override_path, _lm_override_name)
            elif _node_has_lm and _lm_tex_name and _has_lm_uvs:
                lm_img = self._get_tex_by_name(_lm_tex_name)

            # ── Environment map setup (TXI envmaptexture) ──────────────────
            # When TXI defines 'envmaptexture <name>', the diffuse texture alpha
            # channel is the blend weight between the surface colour and the env map.
            # We load the env-map texture here for use in _apply_envmap_to_patch()
            # called per-triangle after the diffuse paste.
            # Note: _apply_kotor_alpha now PRESERVES the alpha channel for env-map
            # textures so the blend weight survives into the rasteriser.
            _node_env_tex_name = str(getattr(node, 'txi_envmaptexture', '')).strip().lower()
            _env_img = self._get_tex_by_name(_node_env_tex_name) if _node_env_tex_name else None


            node_alpha = float(getattr(node, 'alpha', 1.0))
            node_alpha = _clamp(node_alpha, 0.0, 1.0)
            # Full transparency hint pipeline (sourced from xoreos modelnode.cpp):
            #   transparency_hint == 0  → render OPAQUE even if texture has alpha
            #                            (KotOR convention: 0 = opaque/punch-through)
            #   transparency_hint == 1  → TRANSPARENT (alpha-blend, src_alpha blending)
            #   transparency_hint >= 2  → engine-side glass/additive; treat as semi-transparent
            #   beaming == True         → additive glow (handled above via _node_txi_blending=1)
            # The hint only affects default alpha — explicit CTRL_MESH_ALPHA (132) controller
            # values always override it.
            _transp_hint = int(getattr(node, 'transparency_hint', 0))
            # transparency_hint is a render-mode flag ONLY — do NOT force partial
            # alpha from it.  Real glass/additive uses explicit CTRL_MESH_ALPHA (132)
            # or txi_blending flags.  Many KotOR skin meshes have hint=1 but are
            # fully opaque (bump/specular data in DXT5 alpha channel, not transparency).
            # transparency_hint >= 2 is an engine render-mode flag only — do NOT
            # force partial alpha from it.  Real glass/additive uses explicit
            # CTRL_MESH_ALPHA or txi_blending flags.

            # ── Animation overrides: alpha and selfillum from pose ──────────
            # CTRL_MESH_ALPHA (132) and CTRL_MESH_SELFILLUMCOLOR (100) are material
            # controllers that animate per-node opacity and glow independently of
            # skeletal motion.  KotOR uses these heavily for droid eye blinks,
            # glass flickering, and fire/energy FX self-illumination pulses.
            if self._anim_pose is not None:
                _pn_mat = _pose_node_for_transform(node, self._anim_pose)
                if _pn_mat is not None:
                    if _pn_mat.alpha is not None:
                        node_alpha = _clamp(_pn_mat.alpha, 0.0, 1.0)
                    # selfillum will be applied below after 'selfillum' is assigned

            # Base diffuse color fallback
            clean_tex = _clean_tex_name(node.texture)
            if not clean_tex or clean_tex.upper() in ('NULL', ''):
                diff = (0.55, 0.55, 0.65)
            else:
                diff = (
                    _clamp(node.diffuse[0], 0.0, 1.0),
                    _clamp(node.diffuse[1], 0.0, 1.0),
                    _clamp(node.diffuse[2], 0.0, 1.0),
                )

            selfillum = getattr(node, 'selfillum', (0.0, 0.0, 0.0))

            # Apply animated selfillum from pose (CTRL_MESH_SELFILLUMCOLOR=100)
            if self._anim_pose is not None:
                _pn_si = _pose_node_for_transform(node, self._anim_pose)
                if _pn_si is not None and _pn_si.selfillum is not None:
                    selfillum = _pn_si.selfillum

            # ── UV scroll (animate_uv) ─────────────────────────────────────
            # When animate_uv is True, texture scrolls at (uv_dir_x, uv_dir_y)
            # units/sec.  We offset all UVs by current_time * direction each frame.
            # uv_jitter adds a sinusoidal perturbation for water/lava shimmer.
            # Reference: KotorBlender io_scene_kotor/mdl_data.py TrimeshNode fields.
            _node_animate_uv   = bool(getattr(node, 'animate_uv', False))
            _node_uv_dir_x     = float(getattr(node, 'uv_dir_x', 0.0))
            _node_uv_dir_y     = float(getattr(node, 'uv_dir_y', 0.0))
            _node_uv_jitter    = float(getattr(node, 'uv_jitter', 0.0))
            _node_uv_jitter_spd= float(getattr(node, 'uv_jitter_speed', 0.0))
            _node_uv_scroll_u  = 0.0
            _node_uv_scroll_v  = 0.0
            if _node_animate_uv and (_node_uv_dir_x != 0.0 or _node_uv_dir_y != 0.0
                                      or _node_uv_jitter != 0.0):
                _t_anim = getattr(self, '_anim_time', 0.0)
                _node_uv_scroll_u = (_node_uv_dir_x * _t_anim)
                _node_uv_scroll_v = (_node_uv_dir_y * _t_anim)
                if _node_uv_jitter != 0.0 and _node_uv_jitter_spd > 0.0:
                    import math as _muv
                    _jitter = _node_uv_jitter * _muv.sin(_t_anim * _node_uv_jitter_spd * 2.0 * _muv.pi)
                    _node_uv_scroll_u += _jitter
                    _node_uv_scroll_v += _jitter
            # 90° counter-clockwise on the surface.  Implementation: swap U and V
            # and negate the new V: (u, v) → (v, 1.0 - u).
            # Reference: KotorBlender io_scene_kotor reader.py rotatetexture field;
            # xoreos engine source MeshNode::render() UV rotation.
            _node_rotate_tex = bool(getattr(node, 'rotate_texture', False))

            # ── TXI metadata: load and apply texture-specific rendering properties ──
            # TXI (Texture eXtra Info) files provide additional rendering params:
            #   blending=1    → additive blending (glow/fire effects)
            #   blending=2    → punchthrough alpha (hard cutoff)
            #   proceduretype → flipbook animation ('cycle') or water effects
            #   numx/numy/fps → flipbook grid dimensions and speed
            #   clamp_s/t     → UV clamp mode (prevent repeat wrapping)
            _node_txi_blending    = int(getattr(node, 'txi_blending', 0))
            _node_txi_clamp_s     = bool(getattr(node, 'txi_clamp_s', False))
            _node_txi_clamp_t     = bool(getattr(node, 'txi_clamp_t', False))
            # FIX-EDGEBLEED (CPU): Match GPU renderer behaviour — if the node has no
            # explicit TXI repeat/tile setting and all UVs stay within [0,1] (i.e. a
            # UV-atlased character/creature mesh), default to clamp-to-edge on both
            # axes.  This prevents bright corner pixels (e.g. yellow at V≈1.0 of the
            # bantha texture) from bleeding into near-boundary UVs through bilinear
            # interpolation.  Tiling nodes (UVs outside [0,1]) keep GL_REPEAT.
            # Shared single-tile-atlas test (see gpu_diagnostics_records) — keeps
            # this PIL path in lockstep with the accel/GPU paths and fixes the
            # same overshoot-wraps-to-opposite-edge bug.
            if (not _node_txi_clamp_s or not _node_txi_clamp_t) and _node_uses_single_tile_atlas(node):
                _node_txi_clamp_s = True
                _node_txi_clamp_t = True
            # Beaming nodes use additive blending (glow/lightshaft effect).
            # background_geometry nodes (skybox/floor tiles) need no special depth bias —
            # they are sorted naturally by depth, just like opaque geometry.
            # NOTE: beaming overrides txi_blending to additive so the glow composites
            # correctly over the scene regardless of the texture's own TXI settings.
            _node_beaming = bool(getattr(node, 'beaming', False))
            if _node_beaming:
                _node_txi_blending = 1  # treat beaming as additive glow
            _node_txi_procedure   = str(getattr(node, 'txi_proceduretype', ''))
            _node_txi_numx        = int(getattr(node, 'txi_numx', 0))
            _node_txi_numy        = int(getattr(node, 'txi_numy', 0))
            _node_txi_fps         = float(getattr(node, 'txi_fps', 0.0))
            _node_txi_rotate_deg  = float(getattr(node, 'txi_rotate', 0.0))
            # PERF-FIX (v10.2): Pre-compute TXI rotation cos/sin once per node
            # instead of computing them per-face inside the triangle loop.
            if _node_txi_rotate_deg != 0.0:
                _txi_ang = math.radians(_node_txi_rotate_deg * 360.0)
                _txi_ca  = math.cos(_txi_ang)
                _txi_sa  = math.sin(_txi_ang)
            else:
                _txi_ca = _txi_sa = 0.0
            # Is this a flipbook animation?
            _node_is_flipbook = (_node_txi_procedure == 'cycle'
                                 and _node_txi_numx > 0 and _node_txi_numy > 0)
            # Current flipbook frame (based on animation time if available)
            if _node_is_flipbook and _node_txi_fps > 0.0:
                _anim_t = getattr(self, '_anim_time', 0.0)
                _total_frames = _node_txi_numx * _node_txi_numy
                _flip_frame = int(_anim_t * _node_txi_fps) % max(1, _total_frames)
            else:
                _flip_frame = 0

            # Two-sided flag: dangly/cloth nodes and transparency_hint in (1,2) render
            # both faces. KotOR uses this for robes, capes, glass panels, cloth.
            # Also make face/head mesh nodes two-sided to prevent see-through
            # artifacts caused by inner-geometry (eyes, teeth) winding issues.
            transp_hint = getattr(node, 'transparency_hint', 0)
            _nl_tex2 = node.name.lower()
            _is_face_mesh_tex = any(s in _nl_tex2 for s in _FACE_MESH_SUBSTRINGS)
            # BUG FIX v26: same as flat path – inner-geo (eyes, eyelids, teeth)
            # must be two-sided so they aren't back-face culled from outside the head.
            _is_inner_geo_tex2 = any(s in _nl_tex2 for s in _INNER_GEO_SUBSTRINGS)
            is_two_sided = (node.is_dangly
                            or transp_hint in (1, 2)
                            or _is_face_mesh_tex
                            or _is_inner_geo_tex2)

            # ── Inner-geometry tier bump (textured path) ────────────────────
            # Same logic as flat-shade path: eye, eyelid, teeth, and tongue
            # nodes are promoted to tier 1 (drawn after the head/body mesh)
            # so they are revealed through the eye-socket/mouth-gap openings.
            _nl_tex = node.name.lower()
            # BUG FIX v20: same as flat path — removed 'not node.is_skin' gate.
            # Eyeball nodes in K2 head models can be skin nodes; must still promote
            # them to draw tier 1 so they render after the opaque face mesh.
            _clean_tex_ign = _clean_tex_name(getattr(node, 'texture', '') or '')
            _has_tex_ign = bool(_clean_tex_ign and _clean_tex_ign.upper() not in ('NULL', ''))
            _is_inner_geo_tex = (
                _has_tex_ign
                and any(s in _nl_tex for s in _INNER_GEO_SUBSTRINGS)
                and int(transp_hint) == 0
            )

            # Pre-transform ALL vertices to world space (LBS when animated)
            # PERF-FIX (v10.2): Use per-frame vertex/normal cache to avoid
            # redundant transforms across multiple draw passes.
            _node_id = id(node)
            _fvc = getattr(self, '_frame_verts_cache', None)
            _fnc = getattr(self, '_frame_norms_cache', None)
            if _fvc is not None and _node_id in _fvc:
                world_verts = _fvc[_node_id]
            else:
                world_verts = self._get_world_verts_for_node(node)
                if _fvc is not None:
                    _fvc[_node_id] = world_verts

            # Pre-transform normals to world space for correct lighting on rotated nodes
            if _fnc is not None and _node_id in _fnc:
                world_norms = _fnc[_node_id]
            else:
                world_norms = self._get_world_normals_for_node(node)
                if _fnc is not None:
                    _fnc[_node_id] = world_norms
            n_norms = len(world_norms)

            # ── UE-inspired area-weighted vertex normals ──────────────────────
            # Compute smooth per-vertex normals when none are stored in the MDX.
            # Area-weighted accumulation (UE5 SkeletalRenderCPUSkin reference).
            if n_norms == 0 and world_verts and node.faces:
                world_norms = self._compute_area_weighted_normals(node.faces, world_verts)
                n_norms = len(world_norms)

            # Batch-project all world vertices once per node for speed
            screen_verts_t = self._proj_batch(world_verts, W, H)

            # ── Per-node seam-split vertex detection (v10.4 fix) ─────────────
            # Build PER-AXIS sets of vertex indices that are genuine UV-seam-split
            # duplicates: vertices sharing the same 3D position with UV near the
            # OPPOSITE boundary (one near 0, one near 1) on the same axis.
            #
            # WHY PER-AXIS (v10.4b):
            # Using a single combined set incorrectly includes hair-mesh attachment
            # points where several strands start at the same 3D position.  Those
            # positions can have U values of e.g. [0.067, 0.331, 0.912] — near-0
            # and near-1 exist on the U axis — but there is NO V-axis seam at those
            # positions.  If the combined set were used to gate the V-seam fix, it
            # would allow the (erroneous) V-seam fix to run on hair-strand faces,
            # wrapping the V tip vertex outside the texture and producing a black
            # artifact at the tip.
            #
            # SOLUTION: Maintain separate _node_u_seam_verts and _node_v_seam_verts.
            # A face's per-axis skip flag is:
            #   skip_seam_u = not (any vi in _node_u_seam_verts)
            #   skip_seam_v = not (any vi in _node_v_seam_verts)
            # so the V-seam fix is only applied when a genuine V-seam vertex exists.
            #
            # PERF: O(N) positional hash per node, O(1) per-face set lookup.
            # Sets typically contain ≤ 100 vertices for a 1k-vertex mesh.
            _node_u_seam_verts: set = set()
            _node_v_seam_verts: set = set()
            if has_uvs and n_uvs > 0 and node.vertices:
                try:
                    _nv_verts = node.vertices
                    _pos_to_uv_groups: dict = {}
                    # Round to 4 decimal places to handle floating-point imprecision
                    for _vi, (_vpos, _vuv) in enumerate(zip(_nv_verts, uvs)):
                        _pkey = (round(_vpos[0], 4), round(_vpos[1], 4),
                                 round(_vpos[2], 4))
                        if _pkey not in _pos_to_uv_groups:
                            _pos_to_uv_groups[_pkey] = []
                        _pos_to_uv_groups[_pkey].append((_vi, _vuv))
                    # For each position with multiple verts, check axes separately:
                    _SEAM_NEAR = 0.15  # seam vertices within 0.15 of the boundary
                    for _grp in _pos_to_uv_groups.values():
                        if len(_grp) < 2:
                            continue
                        _u_vals = [_uv[0] for _, _uv in _grp]
                        _v_vals = [_uv[1] for _, _uv in _grp]
                        _u_near0 = any(u < _SEAM_NEAR for u in _u_vals)
                        _u_near1 = any(u > 1.0 - _SEAM_NEAR for u in _u_vals)
                        _v_near0 = any(v < _SEAM_NEAR for v in _v_vals)
                        _v_near1 = any(v > 1.0 - _SEAM_NEAR for v in _v_vals)
                        if _u_near0 and _u_near1:
                            for _vi, _ in _grp:
                                _node_u_seam_verts.add(_vi)
                        if _v_near0 and _v_near1:
                            for _vi, _ in _grp:
                                _node_v_seam_verts.add(_vi)
                except Exception:
                    # fallback: treat all verts as non-seam on both axes
                    _node_u_seam_verts = set()
                    _node_v_seam_verts = set()

            for _fi, face in enumerate(node.faces):
                if len(face) < 3:
                    continue
                vi0, vi1, vi2 = face[0], face[1], face[2]
                # Skip degenerate (collapsed) faces with repeated vertex indices
                if vi0 == vi1 or vi1 == vi2 or vi0 == vi2:
                    continue

                # ── Per-face texture (multi-texture mesh support) ──────────
                # For single-texture nodes tex_img is already resolved above.
                # For multi-material nodes (c_bantha body+head, etc.) resolve
                # the correct texture slot for THIS face from face_mats[_fi].
                if _node_is_multitex and has_uvs:
                    _raw_face_tex = self._get_tex_for_face(node, _fi)
                    face_tex = self.tex_cache.get_mip1(_raw_face_tex) if (_use_lq and _raw_face_tex is not None) else _raw_face_tex
                else:
                    face_tex = tex_img
                if vi0 >= nv or vi1 >= nv or vi2 >= nv:
                    continue

                wv0 = world_verts[vi0]
                wv1 = world_verts[vi1]
                wv2 = world_verts[vi2]

                p0 = screen_verts_t[vi0]
                p1 = screen_verts_t[vi1]
                p2 = screen_verts_t[vi2]
                if p0 is None or p1 is None or p2 is None:
                    continue

                # ── Backface culling (screen-space winding order) ────────
                # Screen Y is DOWN; CCW in world (front-facing) = CW in screen
                # → winding cross product is NEGATIVE for front faces.
                # Skip back-facing (winding > 0) in solid mode.
                ex1 = p1[0] - p0[0]; ey1 = p1[1] - p0[1]
                ex2 = p2[0] - p0[0]; ey2 = p2[1] - p0[1]
                winding = ex1 * ey2 - ex2 * ey1
                if winding > 0 and not self.show_wireframe and self.show_solid and not is_two_sided:
                    continue

                # Weighted-centroid depth (average is more stable than min for
                # coplanar/overlapping faces; small face-index bias breaks Z-tie)
                fi_local = len(tris)  # unique per-triangle ID for Z-fight tiebreak
                depth = (p0[2] + p1[2] + p2[2]) * 0.3333 + fi_local * 1e-7

                # ── UVs ─────────────────────────────────────────────────
                if has_uvs:
                    # When face_uvs is populated (ASCII MDL), use tvert indices;
                    # otherwise (binary MDL) use vertex indices directly.
                    if _has_face_uvs:
                        fuv = face_uvs_list[_fi]
                        ti0, ti1, ti2 = fuv[0], fuv[1], fuv[2]
                    else:
                        ti0, ti1, ti2 = vi0, vi1, vi2
                    # Use (0.5, 0.5) fallback for out-of-range tvert indices
                    uv0 = uvs[ti0] if ti0 < n_uvs else (0.5, 0.5)
                    uv1 = uvs[ti1] if ti1 < n_uvs else (0.5, 0.5)
                    uv2 = uvs[ti2] if ti2 < n_uvs else (0.5, 0.5)
                    # ── Lightmap UVs (UV channel 1 / uvs_lm) ────────────────
                    # Binary MDL: lightmap UVs are indexed by vertex index.
                    # Use (0.5,0.5) fallback when lm UVs are absent.
                    if _has_lm_uvs:
                        lm_uv0 = _uvs_lm[vi0] if vi0 < _n_uvs_lm else (0.5, 0.5)
                        lm_uv1 = _uvs_lm[vi1] if vi1 < _n_uvs_lm else (0.5, 0.5)
                        lm_uv2 = _uvs_lm[vi2] if vi2 < _n_uvs_lm else (0.5, 0.5)
                    else:
                        lm_uv0 = lm_uv1 = lm_uv2 = (0.5, 0.5)
                    # rotate_texture: (u,v) → (v, 1-u)  [90° CCW rotation]
                    # Used by KotOR for certain prop nodes (floor decals, lightmapped tiles).
                    if _node_rotate_tex:
                        uv0 = (uv0[1], 1.0 - uv0[0])
                        uv1 = (uv1[1], 1.0 - uv1[0])
                        uv2 = (uv2[1], 1.0 - uv2[0])
                    # ── TXI rotate: additional UV rotation from TXI metadata ──
                    # Some textures have a 'rotate' command in TXI that specifies
                    # an additional rotation angle in turns (0.0–1.0 = 0°–360°).
                    # Apply as a 2D UV rotation around the center (0.5, 0.5).
                    # PERF-FIX (v10.2): _txi_ca/_txi_sa are pre-computed per-node
                    # (not per-face) so no math.cos/sin or closure per triangle.
                    if _node_txi_rotate_deg != 0.0:
                        uu0 = uv0[0]-0.5; vv0 = uv0[1]-0.5
                        uu1 = uv1[0]-0.5; vv1 = uv1[1]-0.5
                        uu2 = uv2[0]-0.5; vv2 = uv2[1]-0.5
                        uv0 = (uu0 * _txi_ca - vv0 * _txi_sa + 0.5,
                               uu0 * _txi_sa + vv0 * _txi_ca + 0.5)
                        uv1 = (uu1 * _txi_ca - vv1 * _txi_sa + 0.5,
                               uu1 * _txi_sa + vv1 * _txi_ca + 0.5)
                        uv2 = (uu2 * _txi_ca - vv2 * _txi_sa + 0.5,
                               uu2 * _txi_sa + vv2 * _txi_ca + 0.5)
                    # ── TXI clamp mode: GL_CLAMP_TO_EDGE when clamp_s/t set ──
                    # Default KotOR wrapping is GL_REPEAT. When the TXI 'clamps'
                    # or 'clampt' command is present, clamp UVs to [0,1] to prevent
                    # texture wrapping artifacts on decals and alpha-blended surfaces.
                    if _node_txi_clamp_s:
                        uv0 = (_clamp(uv0[0], 0.0, 1.0), uv0[1])
                        uv1 = (_clamp(uv1[0], 0.0, 1.0), uv1[1])
                        uv2 = (_clamp(uv2[0], 0.0, 1.0), uv2[1])
                    if _node_txi_clamp_t:
                        uv0 = (uv0[0], _clamp(uv0[1], 0.0, 1.0))
                        uv1 = (uv1[0], _clamp(uv1[1], 0.0, 1.0))
                        uv2 = (uv2[0], _clamp(uv2[1], 0.0, 1.0))
                    if not (
                        bool(getattr(node, "uv_v_flip", True))
                        and bool(getattr(face_tex, "_gr_gpu_uv_v_flip", True))
                    ):
                        # _paste_textured_triangle always applies the KotOR
                        # V flip. Imported DCC meshes and provenance-marked
                        # loose atlases keep bottom-left UVs, so pre-flip here
                        # to cancel that renderer-level flip.
                        uv0 = (uv0[0], 1.0 - uv0[1])
                        uv1 = (uv1[0], 1.0 - uv1[1])
                        uv2 = (uv2[0], 1.0 - uv2[1])
                    # ── TXI flipbook: remap UVs to the current frame cell ─────
                    # KotOR flipbook textures use proceduretype=cycle with numx/numy
                    # to divide the texture into a grid of animation frames.
                    # We remap the face UVs to point within the current frame's cell.
                    if _node_is_flipbook and _node_txi_numx > 0 and _node_txi_numy > 0:
                        uv0 = _compute_flipbook_uv(uv0[0], uv0[1],
                                                    _node_txi_numx, _node_txi_numy, _flip_frame)
                        uv1 = _compute_flipbook_uv(uv1[0], uv1[1],
                                                    _node_txi_numx, _node_txi_numy, _flip_frame)
                        uv2 = _compute_flipbook_uv(uv2[0], uv2[1],
                                                    _node_txi_numx, _node_txi_numy, _flip_frame)
                    # ── UV scroll (animate_uv) ────────────────────────────────────
                    # Add time-based offset to all diffuse UVs.  UVs wrap naturally
                    # (the texture rasteriser uses modulo-1 tiling), so no clamping here.
                    # This replicates the KotOR engine's real-time UV scroll for water,
                    # lava, energy shields, etc.
                    if _node_animate_uv and (_node_uv_scroll_u != 0.0 or _node_uv_scroll_v != 0.0):
                        uv0 = (uv0[0] + _node_uv_scroll_u, uv0[1] + _node_uv_scroll_v)
                        uv1 = (uv1[0] + _node_uv_scroll_u, uv1[1] + _node_uv_scroll_v)
                        uv2 = (uv2[0] + _node_uv_scroll_u, uv2[1] + _node_uv_scroll_v)
                else:
                    uv0 = uv1 = uv2 = (0.5, 0.5)
                    lm_uv0 = lm_uv1 = lm_uv2 = (0.5, 0.5)

                # ── Per-face lighting ────────────────────────────────────
                if n_norms > max(vi0, vi1, vi2):
                    nx = (world_norms[vi0][0] + world_norms[vi1][0] + world_norms[vi2][0]) / 3.0
                    ny = (world_norms[vi0][1] + world_norms[vi1][1] + world_norms[vi2][1]) / 3.0
                    nz = (world_norms[vi0][2] + world_norms[vi1][2] + world_norms[vi2][2]) / 3.0
                    nl = math.sqrt(nx*nx + ny*ny + nz*nz)
                    if nl > 1e-9:
                        nx /= nl; ny /= nl; nz /= nl
                    norm = (nx, ny, nz)
                else:
                    norm = self._face_normal(wv0, wv1, wv2)

                ndotl = _dot(norm, light_dir)
                # Two-sided materials get stronger back-face lighting (cloth/glass)
                ndotl_f = max(0.0, ndotl) + max(0.0, -ndotl) * (0.55 if is_two_sided else 0.35)
                si_r, si_g, si_b = selfillum
                shade = ambient + (1.0 - ambient) * ndotl_f

                # Flat fill color for untextured or fallback
                # face_tex = per-face correct texture (multi-tex) or node tex (single)
                if face_tex is not None:
                    # Shade color for texture modulation (centre sample for fill approx).
                    # UE-inspired mip-bias: use a half-resolution version of the
                    # texture for the centroid colour approximation.  The mip1
                    # image is cached per texture in TextureCache.get_mip1().
                    # This mirrors UE's StreamingManagerTexture mip-level bias:
                    # lower-resolution mips used when per-pixel detail is not required.
                    sample_tex = self.tex_cache.get_mip1(face_tex)
                    uc = (uv0[0] + uv1[0] + uv2[0]) / 3.0
                    vc = (uv0[1] + uv1[1] + uv2[1]) / 3.0
                    tr, tg, tb = self.tex_cache.sample(sample_tex, uc, vc,
                                                        clamp_s=_node_txi_clamp_s,
                                                        clamp_t=_node_txi_clamp_t)
                    # Per-channel Odyssey contract:
                    # texture * (lighting + selfillum) * diffuse tint.
                    dr, dg, db = diff
                    r = int(_clamp(tr * (shade + si_r) * (0.5 + dr*0.5), 0, 255))
                    g = int(_clamp(tg * (shade + si_g) * (0.5 + dg*0.5), 0, 255))
                    b = int(_clamp(tb * (shade + si_b) * (0.5 + db*0.5), 0, 255))
                    fill = (r, g, b)
                else:
                    r = int(_clamp(diff[0] * (shade + si_r) * 255, 0, 255))
                    g = int(_clamp(diff[1] * (shade + si_g) * 255, 0, 255))
                    b = int(_clamp(diff[2] * (shade + si_b) * 255, 0, 255))
                    fill = (r, g, b)

                # shade_color for texture modulation (applied inside _paste_textured_triangle)
                # Per-channel shade colour: diffuse tint preserves model colour
                # while lighting darkens/brightens. Pure grey washes out colour.
                dr, dg, db = diff
                shade_r = int(_clamp(shade * (0.5 + dr*0.5) * 255, 0, 255))
                shade_g = int(_clamp(shade * (0.5 + dg*0.5) * 255, 0, 255))
                shade_b = int(_clamp(shade * (0.5 + db*0.5) * 255, 0, 255))
                shade_col = (shade_r, shade_g, shade_b)

                # Transparent tris (alpha < 1) are appended after opaque so they
                # sort behind opaque geometry at the same depth — correct for glass.
                is_transparent = (node_alpha < 0.999)
                # TXI additive blend (glow/fire) OR beaming nodes: also sorts like transparent
                is_additive = (_node_txi_blending == 1)
                # background_geometry (skybox, floor tiles) should render BEFORE
                # opaque foreground geometry to prevent z-fighting.  We give these a depth
                # BONUS (push them farther away in sort order) so they appear at the bottom
                # of the painter-sort stack (drawn first, overwritten by foreground).
                _node_bg_geom = bool(getattr(node, 'background_geometry', False))
                _bg_bias = 1e-2 if _node_bg_geom else 0.0
                # Use a depth bias so transparent/additive faces sort AFTER opaque at same depth
                sort_depth = depth - (1e-3 if (is_transparent or is_additive) else 0.0) + _bg_bias
                # UE-inspired: convert to sortable uint key for stable integer comparison
                sort_key = _float_to_sort_key(sort_depth)

                # Per-axis seam fix flags.
                #
                # CORE TEXTURE-WRAPPING FIX (v14.1):
                # Previously, a face got skip_seam_u=False (fix applied) ONLY when it
                # contained a vertex in _node_u_seam_verts (a positional duplicate).
                # This was TOO RESTRICTIVE: meshes without positional UV-seam duplicates
                # (non-skin trimeshes, area geometry, creature accessories) would have
                # _node_u_seam_verts = {} → _face_has_u_seam = False → skip_seam_u=True
                # → seam fix NEVER applied → texture stretched across full tile on ALL
                # seam-crossing faces.
                #
                # RULE (v14.1):
                # Let _node_u_seam_verts_found and _node_v_seam_verts_found track
                # whether the analysis ran at all (i.e., whether the mesh had any
                # positional duplicates in either axis).
                #
                #   _node_u_seam_verts is non-empty → u-seam analysis found duplicates;
                #     only fix faces touching a seam vertex (interior faces skipped).
                #   _node_v_seam_verts is non-empty → same for v axis.
                #   BOTH empty AND analysis ran → no duplicates in either axis; let
                #     _paste_textured_triangle's own detection handle both axes.
                #   BOTH empty AND analysis did NOT run (no UVs/verts) → allow both.
                #
                # Hair-strand fix (v10.4b) is preserved:
                #   bthair: _node_u_seam_verts non-empty, _node_v_seam_verts empty.
                #   Because _node_u_seam_verts is non-empty, we know the seam analysis
                #   DID run.  _node_v_seam_verts being empty means there are NO v-seam
                #   duplicates → the V-seam fix stays disabled for all bthair faces.
                #   This prevents the erroneous V-wrap that caused black hair-tip artefacts.
                #
                # For meshes with no duplicates at all (trimesh, area geometry):
                #   Both sets are empty.  The SAFE fast-path inside _paste_textured_triangle
                #   (all UVs in [0.05, 0.95]) covers >80% of faces cheaply.
                #   Only the <20% with seam-crossing UVs are checked by _edge_has_seam.
                if bool(getattr(node, "_external_imported", False)):
                    # External FBX/OBJ/glTF texture atlases are authored in DCC
                    # tools and normally keep every UV island inside 0..1.
                    # KotOR's wrap-seam repair can mistake atlas island borders
                    # for repeating texture seams and smear the wrong part of
                    # the sheet across armor panels, so skip it for imports.
                    _face_has_u_seam = False
                    _face_has_v_seam = False
                else:
                    _any_u_found = bool(_node_u_seam_verts)
                    _any_v_found = bool(_node_v_seam_verts)
                    # Was the analysis actually meaningful? (ran on a mesh with uvs+verts)
                    # We use the presence of either set as evidence that analysis ran and
                    # found at least one axis' worth of duplicates.
                    _analysis_ran = bool(_any_u_found or _any_v_found)

                    if _any_u_found:
                        # Seam analysis found u-duplicates: gate to faces touching a seam vert
                        _face_has_u_seam = (vi0 in _node_u_seam_verts or
                                            vi1 in _node_u_seam_verts or
                                            vi2 in _node_u_seam_verts)
                    else:
                        # Either no u-duplicates found, or analysis found only v-duplicates.
                        # If analysis ran (v-seam found), there are genuinely no u-seam
                        # duplicates → we still need to allow _paste_textured_triangle's
                        # own seam detection for meshes that have seam faces without
                        # positional-duplicate verts (e.g. non-skin area meshes).
                        # Allow seam detection to run (True = don't skip).
                        _face_has_u_seam = True

                    if _any_v_found:
                        # Seam analysis found v-duplicates: gate to faces touching a seam vert
                        _face_has_v_seam = (vi0 in _node_v_seam_verts or
                                            vi1 in _node_v_seam_verts or
                                            vi2 in _node_v_seam_verts)
                    elif _analysis_ran:
                        # Analysis ran (u-seam found) but no v-seam duplicates exist.
                        # This means the mesh genuinely has no V-axis seam faces
                        # (e.g. bthair: u-seam at attachment points but continuous V).
                        # DISABLE v-seam fix to preserve hair-strand black-tip fix (v10.4b).
                        _face_has_v_seam = False
                    else:
                        # Analysis found nothing in either axis: allow both axes to run.
                        _face_has_v_seam = True

                # Two-pass tier: opaque=0, transparent/additive/semi=1.
                # Tier is the PRIMARY sort dimension — all opaque tris are drawn
                # before any transparent tri regardless of depth.  This prevents
                # transparent inner geometry (eyes, droid lenses, glow FX)
                # from rendering on top of opaque face/body meshes when centroid
                # depth ordering alone would place them in front.
                _th_tex = int(getattr(node, 'transparency_hint', 0))
                # Inner-geometry nodes (eyes, eyelids, teeth) are promoted to
                # tier 1 even when transparency_hint==0 so they render AFTER
                # the opaque head/body mesh and are visible through the eye-socket
                # / mouth-gap geometric openings in the face mesh.
                _is_trans_tex = (_th_tex > 0 or is_transparent or is_additive or _is_inner_geo_tex)
                # Draw sky/backdrop panels before every ordinary opaque face.
                # This CPU backend has no depth buffer, so a dedicated tier is
                # required; centroid sorting alone lets a huge sky triangle
                # overwrite closer rooms even though the camera is inside it.
                _tier_tex = -1 if _node_bg_geom else (1 if _is_trans_tex else 0)
                tris.append((sort_key,
                             ((p0[0], p0[1]), (p1[0], p1[1]), (p2[0], p2[1])),
                             fill, shade_col, face_tex, uv0, uv1, uv2, is_sel,
                             fi_local, node_alpha, _node_txi_blending,
                             lm_img, lm_uv0, lm_uv1, lm_uv2,
                             _face_has_u_seam, _face_has_v_seam,
                             _node_txi_clamp_s, _node_txi_clamp_t,
                             _tier_tex))

                if len(tris) >= tri_cap:
                    break
            if len(tris) >= tri_cap:
                break

        # ── Sort: two-pass (tier) then back-to-front (painter's algorithm) ──
        # PRIMARY key: tier (-1=background, 0=opaque, 1=transparent/additive).
        # All opaque triangles render before any transparent triangle
        # regardless of depth.  This prevents transparent inner geometry
        # (eyes, glass, droid lenses) from occluding opaque face/body meshes
        # when centroid-depth ordering alone would place them in front.
        # SECONDARY key: depth (descending = back-to-front within each tier).
        # TERTIARY key: face-insertion index (breaks Z-fighting ties).
        tris.sort(key=lambda t: (t[20], -t[0], t[9]))

        # ── Draw triangles (two-pass: solid first, then wireframe/outlines) ──
        # Pass 1: all solid/texture fills (paste operations modify img in-place)
        wire_tris = []  # collect wireframe data for pass 2
        _mem_error_count = 0  # track consecutive MemoryErrors to abort early
        for entry in tris:
            (depth, pts, fill, shade_col, tex_img, uv0, uv1, uv2, is_sel,
             _fi2, t_alpha, txi_blend, tri_lm_img, lm_uv0, lm_uv1, lm_uv2,
             _tri_face_has_u_seam, _tri_face_has_v_seam,
             _tri_clamp_s, _tri_clamp_t, _tier_draw) = entry
            sp0, sp1, sp2 = pts
            flat = [sp0[0], sp0[1], sp1[0], sp1[1], sp2[0], sp2[1]]

            if self.show_solid:
                if tex_img is not None:
                    # Proper UV-mapped rendering via PIL AFFINE warp.
                    # sel_brightness brightens selected triangles for visual feedback.
                    # node_alpha drives transparency for glass/droid-eye surfaces.
                    # TXI additive blending=1: screen-space additive composite (src+dst).
                    _is_add = (txi_blend == 1)
                    _is_punch = (txi_blend == 2)
                    try:
                        _paste_textured_triangle(
                            img, tex_img,
                            sp0, sp1, sp2,
                            uv0, uv1, uv2,
                            W, H, shade_col,
                            sel_brightness=(50 if is_sel else 0),
                            node_alpha=t_alpha,
                            is_additive=_is_add,
                            skip_seam_u=(not _tri_face_has_u_seam),
                            skip_seam_v=(not _tri_face_has_v_seam),
                            clamp_s=_tri_clamp_s,
                            clamp_t=_tri_clamp_t,
                            is_punchthrough=_is_punch
                        )
                        _mem_error_count = 0  # reset on success
                        # ── Lightmap pass: multiply-blend lightmap over diffuse ──
                        # Only applied to non-additive, non-transparent faces
                        # (additive FX nodes don't have lightmaps in KotOR)
                        if tri_lm_img is not None and not _is_add and t_alpha >= 0.999:
                            _paste_lightmap_triangle(
                                img, tri_lm_img,
                                sp0, sp1, sp2,
                                lm_uv0, lm_uv1, lm_uv2,
                                W, H
                            )
                    except MemoryError:
                        _mem_error_count += 1
                        log.debug(f"_draw_mesh_textured: MemoryError on triangle {_fi2}")
                        if _mem_error_count >= 3:
                            # Too many OOMs in a row — stop textured rendering, fall back
                            log.warning("_draw_mesh_textured: too many MemoryErrors, aborting textured pass")
                            break
                    except Exception:
                        pass  # single-triangle errors are non-fatal
                else:
                    # No texture: flat fill — apply alpha via color blend with background
                    sel_fill = (min(fill[0]+40, 255), min(fill[1]+60, 255), fill[2]) if is_sel else fill
                    if t_alpha < 0.999:
                        # Blend with a mid-grey background for untextured transparent faces
                        bg = (30, 30, 50)
                        a = t_alpha
                        blended = (int(sel_fill[0]*a + bg[0]*(1-a)),
                                   int(sel_fill[1]*a + bg[1]*(1-a)),
                                   int(sel_fill[2]*a + bg[2]*(1-a)))
                        draw.polygon(flat, fill=blended)
                    else:
                        draw.polygon(flat, fill=sel_fill)

            if self.show_wireframe or is_sel:
                wire_col = _SEL[:3] if is_sel else _WIRE[:3]
                wire_tris.append((flat, wire_col))

        # Pass 2: wireframe / selection outlines (with fresh draw context after paste)
        if wire_tris:
            draw = ImageDraw.Draw(img)
            for flat, wire_col in wire_tris:
                draw.polygon(flat, outline=wire_col)

        # NOTE: _draw_bones is called by render() with a fresh draw context.

    def _is_deformation_helper(self, node: 'ModelNode') -> bool:
        """
        Detect KotOR deformation-helper mesh nodes that should NOT be rendered
        as visible geometry.

        OBJ / FBX imported nodes are tagged with node._imported = True by
        OBJImporter and FBXImporter.  These are never deformation helpers —
        they are real geometry the user explicitly loaded.

        In KotOR's Odyssey engine, character models contain hidden deformation
        helper trimeshes (usually ending in _g, _G, or matching bone names like
        lbicep_g, rthigh_g, pelvis_g, head_g, jaw2, etc.).  These are used by
        the engine's SkinMesh deformation pipeline and are never rendered directly.
        They have:
          - No texture (tex=null/empty) or helper-style node naming
          - Often named with a _g / _G suffix (geometry deformation)
          - Sometimes carry a visible texture name but with completely invalid UVs
            or NO UVs at all
          - Helper-style names are only a fallback when the node does not carry
            explicit visible-surface evidence

        IMPORTANT: Skin nodes with a real (non-null) texture AND valid UVs are
        ALWAYS renderable geometry, even if their name ends in _g.  Some KotOR
        models (e.g. n_darthrevanm, n_darthrevanf, p_bastilabb02) use _g-named
        skin meshes as their primary visible geometry.

        Some KOTOR 2 droids use non-skin _g nodes for their visible articulated
        body.  Those nodes are renderable when the MDL render flag is true and
        they carry both a real diffuse texture and UV coordinates.  Human rig
        helpers such as n_admrlsaulkar remain hidden because render is false.

        v12.14: Also treats skin-proxy nodes (identified by _compute_skin_proxy_ids)
        as deformation helpers.  A non-skin node is a proxy when it shares an
        exclusive texture with exactly one skin mesh that has more vertices
        (e.g. 'head_Hair' on c_bantha is a 61-vert proxy for 'bthair' with 320 verts).
        """
        # ── OBJ / FBX imported nodes: always renderable ───────────────────────
        # Nodes tagged with _imported=True were explicitly loaded by the user from
        # an OBJ or FBX file.  They are never KotOR deformation helpers — skip all
        # KotOR-specific heuristics and render them unconditionally.
        if getattr(node, '_imported', False):
            return False

        tex = _clean_tex_name(node.texture)
        is_null_tex = (not tex or tex.upper() == 'NULL')

        # ── BUG FIX v26: Inner-geometry nodes (eyes, eyelids, teeth, tongue, ─
        # jaw, gum) are ALWAYS renderable when they have a real texture and
        # UVs — regardless of is_skin status, name suffix, or proxy rules.
        # These nodes sit inside the face mesh and form the visible eye/mouth
        # content.  Treating them as deformation helpers (for any reason) causes
        # them to be silently dropped from the render list and the character
        # appears eyeless.  This explicit early-return short-circuits ALL later
        # helper checks, including the _skin_proxy_ids check.
        _name_lower_check = node.name.lower()
        if any(s in _name_lower_check for s in _INNER_GEO_SUBSTRINGS):
            if not is_null_tex and node.uvs:
                return False  # always render inner-geo nodes

        # ── Skin node with a real texture and UVs → always visible ────────────
        # Never treat it as a deformation helper regardless of name.
        # (Some KotOR models use _g-named skin meshes as primary geometry.)
        if node.is_skin and not is_null_tex and node.uvs:
            return False

        # ── Explicit visible rigid surface → always visible ──────────────────
        # K2 construction/mining droids use _g names for authored rigid body
        # segments, not deformation proxies.  Respect the MDL's material and
        # render evidence before applying the legacy suffix fallback.
        if _node_has_renderable_rigid_surface(node):
            return False

        # ── Remaining non-skin _g / _G or _dum nodes are deform helpers ──────
        # At this point imported meshes, inner geometry, skinned surfaces, and
        # explicitly visible rigid surfaces have all been accepted.
        name_lower = node.name.lower()
        _name_is_inner_geo = any(s in name_lower for s in _INNER_GEO_SUBSTRINGS)
        if not node.is_skin and (name_lower.endswith('_g')
                                  or name_lower.endswith('_g0')
                                  or name_lower.endswith('_dum')):
            # EXCEPTION: inner-geometry nodes with a real texture and UVs are
            # ALWAYS renderable — they are real eyeball /
            # teeth / tongue meshes, not deformation proxies.
            if _name_is_inner_geo and not is_null_tex and node.uvs:
                return False  # render this inner-geo node
            return True

        # ── Null-texture, non-skin nodes → always deform helpers ─────────────
        if is_null_tex and not node.is_skin:
            return True

        # ── Null-texture skin nodes with no UVs or only zero UVs → helpers ───
        if is_null_tex and node.is_skin and (not node.uvs
                            or all(u == 0.0 and v == 0.0
                                   for u, v in node.uvs[:5])):
            return True

        # ── Non-skin nodes with NO UVs → deform helpers UNLESS module/area model ──
        # KotOR creature/character models contain skeleton-bone helper nodes
        # (e.g. BTHips, BTSpine1, BTHead, BTShoulders on the bantha; or similar
        # bone-proxy trimeshes on other creatures) that:
        #   - Carry a real texture name (e.g. 'c_bantha01') but NO UV coords
        #   - Are NOT skin nodes (is_skin=False)
        #   - Are NOT named _g / _dum (so the suffix check above doesn't catch them)
        # These are the raw bone geometry that the engine uses internally for
        # collision/deformation but never renders directly.  Without UVs they
        # cannot be textured, and rendering them as flat-shaded produces ugly
        # opaque bone-shaped blobs that obscure the real skin mesh.
        #
        # EXCEPTION: Module/area models (classification 'effect'=0 or 'tile'=2)
        # store ALL vertex data (including UVs) in the companion .mdx file.
        # When the MDX is not available or not yet loaded, UV arrays are empty but
        # the geometry IS real renderable geometry.  We must NOT discard it.
        # For these models, render even without UVs (flat-shaded fallback).
        # Also: AABB nodes in room models are always real geometry (walkmesh).
        if not node.is_skin and not node.uvs:
            model_cls = getattr(self.model, 'classification', 'character') if self.model else 'character'
            model_type = getattr(self.model, 'model_type', 4) if self.model else 4
            # Module/area/effect models: render all non-_g trimeshes even without UVs
            if model_cls in ('effect', 'tile', 'other') or model_type in (0, 2):
                # Still skip obvious _g deformation proxies
                if not (name_lower.endswith('_g') or name_lower.endswith('_g0') or name_lower.endswith('_dum')):
                    return False  # render as flat-shaded geometry
            return True

        # ── v12.14: Skin-proxy detection (non-skin node with exclusive-texture ──
        # skin-mesh counterpart).  E.g. 'head_Hair' on c_bantha (61 verts, c_banthh01)
        # is a deformation proxy for 'bthair' (320 verts, c_banthh01).
        _proxy_ids = getattr(self, '_skin_proxy_ids', None)
        if _proxy_ids is not None and id(node) in _proxy_ids:
            return True

        return False

    def _iter_visible_mesh_nodes(self):
        """Yield mesh nodes that have visible geometry (not deform helpers or outlier proxies).

        Dangly (cloth) nodes are ALWAYS rendered — they represent visible cloth
        geometry even if they share properties with deformation helpers.

        BUG-C FIX: Respect the KotOR MDL 'render' flag.  Nodes with render=False
        are explicitly marked as invisible by the model author and must never be
        drawn.  Previously these nodes were rendered despite the flag, causing
        invisible geometry bleed (e.g. collision proxy meshes appearing as solid
        black patches over the visible model).

        v12.15 FIXES (from deep research):
        - Skip SABER nodes (NODE_SABER=0x0800): lightsaber blade geometry is
          runtime-generated by the engine; the saber mesh node only provides anchor
          positions.  Rendering it produces garbage quad geometry.
        - Skip attachment/hook dummy nodes by well-known naming conventions:
          camerahook, headhook, handhook_*, handconjure*, etc.
          These are empty DUMMY nodes used as VFX/camera attachment points.
        """
        # Attachment/hook name prefixes — these are always non-renderable dummy nodes
        # used as VFX, camera, and weapon attachment points in KotOR character models.
        _HOOK_PREFIXES = (
            'camerahook', 'headhook', 'handhook',
            'handconjure', 'chestconjure', 'footstep', 'impact_', 'ap_',
        )
        if is_animation_supermodel(self.model):
            return
        for n in self._iter_mesh_nodes():
            if getattr(n, '_gr_hidden', False):
                continue
            # Skip nodes explicitly marked non-renderable by the MDL author.
            # The render flag is set to False for collision boxes, occluders, and
            # internal engine helpers.  Always respect it regardless of other flags.
            # Exception: selected node is always shown for editing purposes.
            # EXCEPTION: inner-geometry nodes (eyes, eyelids, teeth, tongue) are
            # always rendered even if render=0 — some KotOR NPC head MDLs store
            # render=0 on eyeball/teeth nodes which would make the face appear empty.
            _nl_tex = n.name.lower()
            _is_inner_geo_tex = any(s in _nl_tex for s in _INNER_GEO_SUBSTRINGS)
            if not getattr(n, 'render', True) and n is not self.selected_node and not _is_inner_geo_tex:
                continue
            # Skip SABER nodes — lightsaber blade is procedurally generated
            # at runtime.  The node only provides anchor/extent information; rendering
            # it as geometry produces degenerate quads.
            if getattr(n, 'is_saber', False):
                continue
            # Skip attachment/hook dummy nodes by name convention.  Do not use
            # broad ``lhand``/``rhand`` prefixes here: droid models use names
            # such as LHand2, LHandDrd, and lhand_g for real visible meshes.
            _nl = n.name.lower()
            if any(_nl.startswith(pfx) or _nl == pfx for pfx in _HOOK_PREFIXES):
                continue
            # Dangly/cloth nodes are never deformation helpers — always render them
            if n.is_dangly:
                yield n
                continue
            if not self._is_deformation_helper(n) and not self._is_outlier_skin(n):
                yield n
