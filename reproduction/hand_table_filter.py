"""Replay-only isolation of hand/table pairs, preserving other filter pairs."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

FIELDS = ['filter', 'friction', 'rolling_friction', 'torsion_friction',
          'restitution', 'compliance', 'contact_offset', 'rest_offset', 'thickness']


def apply(task, output, disabled):
    root = Path(__file__).resolve().parents[1]
    urdf = root / 'dex4d_policy/assets' / task.cfg['env']['asset']['assetFileName']
    links = {'palm_lower'}
    joints = ET.fromstring(urdf.read_text()).findall('joint')
    while True:
        expanded = links | {j.find('child').attrib['link'] for j in joints
                            if j.find('parent').attrib['link'] in links}
        if expanded == links:
            break
        links = expanded
    rows = []
    for env in task.envs:
        handles = {name: task.gym.find_actor_handle(env, name) for name in ['robot', 'object', 'table']}
        records, labels, before, after = {}, [], [], []
        robot = handles['robot']
        body_names = task.gym.get_actor_rigid_body_names(env, robot)
        ranges = task.gym.get_actor_rigid_body_shape_indices(env, robot)
        hand = [name for name, span in zip(body_names, ranges) if name in links and span.count]
        assert len(hand) == 17, hand
        bits = {name: 1 << i for i, name in enumerate(hand)}
        for actor_name, handle in handles.items():
            props = task.gym.get_actor_rigid_shape_properties(env, handle)
            old = [{k: getattr(p, k) for k in FIELDS} for p in props]
            assert all(p['filter'] == 0 for p in old), 'Diagnostic expects original zero masks'
            names = task.gym.get_actor_rigid_body_names(env, handle)
            spans = task.gym.get_actor_rigid_body_shape_indices(env, handle)
            shape_bodies = [None] * len(props)
            for name, span in zip(names, spans):
                for i in range(span.start, span.start + span.count):
                    shape_bodies[i] = name
                    if disabled:
                        props[i].filter = (bits.get(name, 0) if actor_name == 'robot'
                                           else sum(bits.values()) if actor_name == 'table' else 0)
            assert all(name is not None for name in shape_bodies)
            if disabled and actor_name != 'object':
                assert task.gym.set_actor_rigid_shape_properties(env, handle, props)
            new = [{k: getattr(p, k) for k in FIELDS}
                   for p in task.gym.get_actor_rigid_shape_properties(env, handle)]
            for x, y, expected in zip(old, new, props):
                assert all(x[k] == y[k] for k in FIELDS if k != 'filter')
                assert y['filter'] == expected.filter
            records[actor_name] = {'shape_bodies': shape_bodies, 'before': old, 'after': new}
            labels.extend((actor_name, name) for name in shape_bodies)
            before.extend(p['filter'] for p in old)
            after.extend(p['filter'] for p in new)
        a, b = np.asarray(before, dtype=np.int64), np.asarray(after, dtype=np.int64)
        changed = ((a[:, None] & a) == 0) != ((b[:, None] & b) == 0)
        desired = np.zeros_like(changed)
        for i, (actor, body) in enumerate(labels):
            for j, (other_actor, other_body) in enumerate(labels):
                if actor == other_actor and body == other_body:
                    changed[i, j] = False  # Shapes of one rigid body never self-collide.
                desired[i, j] = disabled and (
                    (actor == 'robot' and body in bits and other_actor == 'table') or
                    (other_actor == 'robot' and other_body in bits and actor == 'table'))
        assert np.array_equal(changed, desired), 'Unexpected collision pair change'
        rows.append({'actors': records, 'changed_unordered_shape_pairs': int(changed.sum() // 2)})
    report = {'hand_table_contact_disabled': disabled, 'hand_body_bits': bits,
              'environments': rows, 'all_other_filter_pairs_preserved': True,
              'nonfilter_shape_properties_exact': True,
              'scope': 'Only selected hand/table shape pairs change filter eligibility. Same-body shape pairs excluded as nonphysical; object/table, hand/object, arm/table and distinct-body self pairs retain eligibility. Native primitive calibration checks API behavior separately.'}
    (output / 'hand_table_filter.json').write_text(json.dumps(report, indent=2) + '\n')
