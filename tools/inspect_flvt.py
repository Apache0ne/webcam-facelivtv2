#!/usr/bin/env python3
import argparse
import struct

HEADER_FMT="<8sIIQQQI20s"
DESC_FMT="<96sII4IQQQ"
HS=struct.calcsize(HEADER_FMT)
DS=struct.calcsize(DESC_FMT)

ap=argparse.ArgumentParser()
ap.add_argument("model")
a=ap.parse_args()
with open(a.model,"rb") as f:
    magic,ver,n,toff,doff,dbytes,dtype,_=struct.unpack(HEADER_FMT,f.read(HS))
    print("magic",magic,"version",ver,"tensors",n,"data_bytes",dbytes,"dtype",dtype)
    f.seek(toff)
    for _ in range(n):
        d=struct.unpack(DESC_FMT,f.read(DS))
        name=d[0].split(b"\0",1)[0].decode("utf-8")
        dtype,ndim=d[1],d[2]
        dims=tuple(d[3:7][:ndim])
        off,nbytes=d[7],d[8]
        print(f"{name:52s} dtype={dtype} shape={str(dims):22s} off={off:10d} bytes={nbytes:10d}")
