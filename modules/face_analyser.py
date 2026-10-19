from typing import Any, List, Optional
import insightface
import numpy as np

import modules.globals
from modules.typing import Frame,Face
from modules.accel import get_providers

FACE_ANALYSER = None


def get_face_analyser() -> Any:
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        initialize_face_analyser()
    return FACE_ANALYSER




def get_one_face(frame: Frame) -> Optional[Face]:
    faces = FACE_ANALYSER.get(frame, max_num=1)
    return faces[0] if faces else None





def get_many_faces(frame: Frame) -> List[Face]:
    return FACE_ANALYSER.get(frame)










def initialize_face_analyser():
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        # landmark_3d_68 and genderage are never used by the app, so skip loading/running them
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=get_providers(),
                                                     allowed_modules=['detection', 'recognition', 'landmark_2d_106'])
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))

def get_many_faces_fast(frame: Frame, with_embedding: bool = False) -> List[Face]:
    """
    Live-path detection: detector + 106 landmarks only. The 512-d recognition embedding is
    only computed when asked for (face tracking needs it, a plain swap does not).
    """
    analyser = get_face_analyser()
    bboxes, kpss = analyser.det_model.detect(frame, max_num=0, metric='default')
    faces = []
    for i in range(bboxes.shape[0]):
        face = Face(bbox=bboxes[i, 0:4], kps=kpss[i] if kpss is not None else None, det_score=bboxes[i, 4])
        analyser.models['landmark_2d_106'].get(frame, face)
        if with_embedding:
            analyser.models['recognition'].get(frame, face)
        faces.append(face)
    return faces

def get_one_face_left(frame: Frame) -> Optional[Face]:
    faces = FACE_ANALYSER.get(frame)
    return min(faces, key=lambda x: x.bbox[0]) if faces else None

def get_one_face_right(frame: Frame) -> Optional[Face]:
    faces = FACE_ANALYSER.get(frame)
    return max(faces, key=lambda x: x.bbox[0]) if faces else None

def get_two_faces(frame: Frame) -> List[Face]:
    faces = FACE_ANALYSER.get(frame, max_num=2)
    return sorted(faces, key=lambda x: x.bbox[0])

def prepare_source_faces(faces: List[Face], average: Optional[bool] = None) -> List[Face]:
    """
    With 'average source faces' on, all faces in the source image are treated as photos of the
    same person and merged into one identity (mean of their normalised embeddings). Several
    angles/lighting conditions give a steadier, more accurate likeness than a single photo.
    """
    average = modules.globals.average_source_faces if average is None else average
    if not average or len(faces) < 2:
        return faces
    embedding = np.mean([face.normed_embedding for face in faces], axis=0)
    norm = np.linalg.norm(embedding)
    merged = Face(dict(max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))))
    merged.embedding = embedding / norm * np.linalg.norm(merged.embedding)
    return [merged]
