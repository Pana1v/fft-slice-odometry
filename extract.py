"""Stream LiDAR + ground truth out of an NTU VIRAL zip, even a truncated download.

zip entry (raw deflate) -> ROS1 bag records -> chunks -> messages.
Only the two Ouster topics and the Leica pose are kept, so the 4.3 GB bag
never touches the disk.

Usage: python extract.py eee_03.zip out_dir
"""
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
from rosbags.typesys import Stores, get_typestore

LIDAR_H = "/os1_cloud_node1/points"
LIDAR_V = "/os1_cloud_node2/points"
GT_TOPIC = "/leica/pose/relative"

# Lidar -> body, from brytsknguyen/SLICT config/ntuviral.yaml
EXTR = {
    LIDAR_H: (np.eye(3), np.array([-0.050, 0.000, 0.055])),
    LIDAR_V: (np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
              np.array([-0.550, 0.030, 0.090])),
}
MIN_RANGE = 0.8      # drops zero returns and the airframe itself
MAX_RANGE = 40.0
PAIR_TOL = 0.05      # s, horizontal/vertical scans are merged within this gap

OP_MSG, OP_CHUNK, OP_CONN = 0x02, 0x05, 0x07
READ_BLOCK = 16 << 20


def bag_stream(zip_path):
    """Yield decompressed bytes of the first .bag entry, stopping cleanly at truncation."""
    f = open(zip_path, "rb")
    pos = 0
    while True:
        f.seek(pos)
        h = f.read(30)
        if h[:4] != b"PK\x03\x04":
            raise ValueError(f"no .bag entry found in {zip_path}")
        csz, nl, xl = struct.unpack_from("<I", h, 18)[0], *struct.unpack_from("<HH", h, 26)
        name = f.read(nl).decode()
        if name.endswith(".bag"):
            f.seek(pos + 30 + nl + xl)
            break
        pos += 30 + nl + xl + csz

    d = zlib.decompressobj(-15)
    while block := f.read(READ_BLOCK):
        yield d.decompress(block)


def fields(buf):
    out, q = {}, 0
    while q < len(buf):
        n = struct.unpack_from("<I", buf, q)[0]
        k, v = buf[q + 4:q + 4 + n].split(b"=", 1)
        out[k.decode()] = v
        q += 4 + n
    return out


def records(buf):
    """Parse complete records from buf; return (records, bytes consumed)."""
    out, p = [], 0
    while p + 4 <= len(buf):
        hl = struct.unpack_from("<I", buf, p)[0]
        if p + 8 + hl > len(buf):
            break
        dl = struct.unpack_from("<I", buf, p + 4 + hl)[0]
        end = p + 8 + hl + dl
        if end > len(buf):
            break
        out.append((fields(buf[p + 4:p + 4 + hl]), buf[p + 8 + hl:end]))
        p = end
    return out, p


def cloud_xyz(msg):
    """PointCloud2 -> (N,3) float32 using the x/y/z field offsets."""
    off = {f.name: f.offset for f in msg.fields}
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
    xyz = np.stack([raw[:, off[k]:off[k] + 4].copy().view(np.float32)[:, 0] for k in "xyz"], 1)
    return xyz


def main():
    zip_path, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    ts = get_typestore(Stores.ROS1_NOETIC)

    conns = {}                 # conn id -> (topic, msgtype)
    clouds = {LIDAR_H: [], LIDAR_V: []}
    gt = []
    buf = b""
    started = False

    for piece in bag_stream(zip_path):
        buf += piece
        if not started:
            assert buf[:13] == b"#ROSBAG V2.0\n", buf[:13]
            buf, started = buf[13:], True

        recs, used = records(buf)
        buf = buf[used:]
        for hdr, data in recs:
            if hdr["op"][0] != OP_CHUNK:
                continue
            assert hdr["compression"] == b"none", hdr["compression"]
            for h2, d2 in records(data)[0]:
                op = h2["op"][0]
                cid = struct.unpack("<I", h2["conn"])[0] if "conn" in h2 else None
                if op == OP_CONN:
                    typ = fields(d2)["type"].decode().replace("/", "/msg/")
                    conns[cid] = (h2["topic"].decode(), typ)
                    continue
                if op != OP_MSG or conns.get(cid, ("",))[0] not in (LIDAR_H, LIDAR_V, GT_TOPIC):
                    continue

                topic, typ = conns[cid]
                msg = ts.deserialize_ros1(d2, typ)
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                if topic == GT_TOPIC:
                    p = msg.pose.position
                    gt.append((stamp, p.x, p.y, p.z))
                    continue

                xyz = cloud_xyz(msg)
                r = np.linalg.norm(xyz, axis=1)
                xyz = xyz[(r > MIN_RANGE) & (r < MAX_RANGE)]
                R, t = EXTR[topic]
                clouds[topic].append((stamp, (xyz @ R.T + t).astype(np.float32)))

        n_h = len(clouds[LIDAR_H])
        if n_h and n_h % 200 == 0:
            print(f"  {n_h} scans, {len(gt)} gt, t={clouds[LIDAR_H][-1][0] - clouds[LIDAR_H][0][0]:.1f} s",
                  flush=True)

    # Merge each horizontal scan with the nearest vertical scan
    v_st = np.array([s for s, _ in clouds[LIDAR_V]])
    stamps, merged, n_h = [], [], []
    for s, ph in clouds[LIDAR_H]:
        k = int(np.argmin(np.abs(v_st - s)))
        if abs(v_st[k] - s) > PAIR_TOL:
            continue
        stamps.append(s)
        n_h.append(len(ph))
        merged.append(np.concatenate([ph, clouds[LIDAR_V][k][1]]))

    offsets = np.r_[0, np.cumsum([len(m) for m in merged])]
    np.savez(out / "frames.npz", stamps=np.array(stamps), offsets=offsets,
             points=np.concatenate(merged), n_h=np.array(n_h))

    gt = np.array(gt)
    rows = np.c_[gt, np.zeros((len(gt), 3)), np.ones(len(gt))]   # position-only GT
    np.savetxt(out / "gt_prism.tum", rows, fmt="%.9f")

    dt = np.diff(stamps)
    print(f"frames {len(stamps)} over {stamps[-1] - stamps[0]:.1f} s (dt median {np.median(dt)*1e3:.1f} ms, "
          f"max {dt.max()*1e3:.1f}), pts/frame {np.mean(np.diff(offsets)):.0f}, gt {len(gt)}")
    print(f"topics seen: {sorted(t for t, _ in conns.values())}")


if __name__ == "__main__":
    main()
