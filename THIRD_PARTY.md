# Third-party reference

The runtime architecture and checkpoint converter target the official FaceLiVT / FaceLiVTv2 model definition supplied by the user from the FaceLiVT project. The upstream project license is reproduced in `third_party/FaceLiVT_LICENSE_BSD3.txt`.

The repository's shared model artifact is `facelivtv2-l.fp16.flvt`, converted
from the `facelivtv2-l.pt` checkpoint supplied by the project owner. The
BSD-3-Clause license in this folder applies to the upstream code; it does not
by itself establish redistribution rights for the checkpoint or its converted
weights. Confirm the model's terms before publishing the `.flvt` artifact.

## Optional webcam detector

SCRFD architecture, preprocessing, and landmark conventions are described by
the upstream InsightFace project:
https://github.com/deepinsight/insightface/tree/master/detection/scrfd
and https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md.
The downloaded SCRFD-10G weights are separate from the native FaceLiVT model.
InsightFace's public pretrained weights are provided for non-commercial research;
commercial use requires separate licensing from their provider. This restriction
concerns the weights, not merely the code license. See the upstream model zoo for
the applicable terms. WEBCAM.md records the detector download and checksum.

OpenCV is used for small CPU-side image-quality and display operations. Video
capture/display use the Windows Media Foundation and D3D11/CUDA path. The webcam
frontend does not use SFace/ArcFace recognition weights or the InsightFace
Python package.
