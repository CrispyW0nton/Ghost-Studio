"""Project editable Unreal components into the qualified Odyssey sector policy."""
from __future__ import annotations

import copy
from dataclasses import replace

from src.math.mesh_instance_transform import bake_mesh_instance
from .authored_imported_mesh import ImportedMeshRoomPrimitive
from .authored_module_walkmesh import offset_wok_data


def project_unreal_sector(project):
    contract = project.extra.get("unreal_import", {})
    if contract.get("room_partition") != "independent-components-v1":
        return project
    rooms = project.rooms
    if any(not isinstance(r.primitive, ImportedMeshRoomPrimitive) or r.composition is not None for r in rooms):
        raise ValueError("Unreal sector compilation requires explicit imported mesh components; bake added primitives first.")
    playable = [r for r in rooms if r.primitive.wok and r.primitive.wok.faces]
    if len(playable) != 1:
        raise ValueError("Unreal v1 sector compilation requires one connected authored walkmesh.")
    floor = playable[0]
    nodes, sources, surfaces, resources = {}, [], [], {}
    for index, room in enumerate(rooms):
        identity = room.metadata.get("source_component_id", room.room_resref)
        names = []
        for surface in room.primitive.surfaces:
            name = f"c{index:02d}_{surface.name}"
            names.append(name)
            if any(room.position):
                vertices, normals, faces = bake_mesh_instance(surface.vertices, surface.normals, surface.faces,
                    position=room.position, rotation=(0, 0, 0, 1))
                surface = replace(surface, vertices=tuple(map(tuple, vertices)), normals=tuple(map(tuple, normals)), faces=tuple(map(tuple, faces)))
            surfaces.append(replace(surface, name=name))
        nodes[identity] = names
        resources[identity] = floor.room_resref
        sources.append({**copy.deepcopy(room.metadata), "source_room_resref": room.room_resref, "sector_nodes": names})
    metadata = {**copy.deepcopy(floor.metadata), "room_partition": "single-playable-sector-v1", "source_components": sources}
    wok = offset_wok_data(floor.primitive.wok, floor.position) if any(floor.position) else copy.deepcopy(floor.primitive.wok)
    sector = replace(floor, position=(0, 0, 0), visible_rooms=(), metadata=metadata,
                     primitive=replace(floor.primitive, surfaces=tuple(surfaces), wok=wok, metadata=metadata))
    extra = copy.deepcopy(project.extra)
    extra["unreal_import"].update(room_partition="single-playable-sector-v1", component_nodes=nodes, component_resources=resources)
    lights = tuple(replace(light, room_resref=floor.room_resref) for light in project.lights)
    return replace(project, rooms=(sector,), lights=lights, extra=extra)
