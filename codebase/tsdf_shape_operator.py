"""Fixed-frame TSDF measurement for Shape-QD parity checks."""
from __future__ import annotations
import numpy as np
import trimesh


def lattice(bounds, resolution):
    lo, hi = np.asarray(bounds[0],float), np.asarray(bounds[1],float)
    axes=[np.linspace(lo[i],hi[i],resolution,endpoint=True) for i in range(3)]
    xyz=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
    return xyz, lo, hi


def mesh_tsdf(mesh_path, bounds, resolution=96, truncation_m=.012, chunk=100000):
    """World-frame truncated signed-distance grid, positive inside trimesh."""
    mesh=trimesh.load(mesh_path,force='mesh',process=False)
    if isinstance(mesh,trimesh.Scene): mesh=trimesh.util.concatenate(tuple(mesh.dump()))
    xyz,_,_=lattice(bounds,resolution)
    try:
        from pysdf import SDF
        field=SDF(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
        query=lambda q: -field(np.ascontiguousarray(q,dtype=np.float32), n_threads=4)
    except ImportError:
        query=lambda q: trimesh.proximity.signed_distance(mesh,q)
    values=[]
    for start in range(0,len(xyz),chunk):
        values.append(query(xyz[start:start+chunk]))
    sdf=np.concatenate(values).reshape((resolution,)*3)
    return np.clip(sdf/truncation_m,-1.,1.).astype(np.float32)


def masked_multiscale_tsdf_loss(field, target, mask=None):
    """Numpy diagnostic equivalent of the planned differentiable multi-scale Huber loss."""
    import scipy.ndimage as ndi
    total=0.
    for factor,weight in ((1,.5),(2,1.),(4,2.),(8,3.)):
        a=field if factor==1 else ndi.uniform_filter(field,size=factor)[::factor,::factor,::factor]
        b=target if factor==1 else ndi.uniform_filter(target,size=factor)[::factor,::factor,::factor]
        if mask is not None:
            m=mask if factor==1 else ndi.uniform_filter(mask.astype(float),size=factor)[::factor,::factor,::factor]>.5
            d=np.abs(a[m]-b[m])
        else: d=np.abs(a-b).ravel()
        total += weight*np.mean(np.where(d<.1,.5*d*d,.1*(d-.05)))
    return float(total)
