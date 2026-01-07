from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
from insightface.app import FaceAnalysis
import os
import numpy as np
from typing import List
from fastapi import UploadFile, File


app = FastAPI()
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

@app.post("/debug/decode")
async def debug_decode(image: UploadFile = File(...)):
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image (OpenCV returned None).")

    h, w = frame.shape[:2]
    target_w = 640
    if w > target_w:
        scale = target_w / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))


@app.post("/enroll")
async def enroll(user_id: str, images: List[UploadFile] = File(...)):
    if len(images) < 3:
        raise HTTPException(status_code=400, detail="Provide at least 3 images.")

    embs = []
    for img_file in images:
        data = await img_file.read()
        img = decode_to_bgr(data)
        if img is None:
            continue
        emb = get_embedding(img)
        if emb is not None:
            embs.append(emb)

    if len(embs) < 2:
        raise HTTPException(status_code=400, detail="Not enough usable faces found.")

    avg_emb = np.mean(np.stack(embs, axis=0), axis=0).astype(np.float32)
    out_path = os.path.join("data", "embeddings", f"{user_id}.npy")
    np.save(out_path, avg_emb)

    return {"enrolled": True, "user_id": user_id, "used_images": len(embs)}

@app.post("/verify")
async def verify(image: UploadFile = File(...), user_id: str = "rahum", threshold: float = 0.35):
    path = os.path.join("data", "embeddings", f"{user_id}.npy")
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail=f"No enrolled user '{user_id}'. Enroll first.")

    enrolled_emb = np.load(path).astype(np.float32)

    data = await image.read()
    img = decode_to_bgr(data)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    emb = get_embedding(img)
    if emb is None:
        return {"status": "NO_FACE"}

    score = cosine_similarity(emb, enrolled_emb)
    status = "APPROVED" if score >= threshold else "DENIED"
    return {"status": status, "score": score, "threshold": threshold, "user_id": user_id}


def decode_to_bgr(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def largest_face(faces):
    if not faces:
        return None
    def area(f):
        x1, y1, x2, y2 = f.bbox
        return float((x2 - x1) * (y2 - y1))
    return max(faces, key=area)

def get_embedding(img_bgr):
    faces = face_app.get(img_bgr)
    f = largest_face(faces)
    if f is None:
        return None
    return f.embedding.astype(np.float32)

def decode_to_bgr(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def largest_face(faces):
    if not faces:
        return None
    def area(f):
        x1, y1, x2, y2 = f.bbox
        return float((x2 - x1) * (y2 - y1))
    return max(faces, key=area)

def get_embedding(img_bgr):
    faces = face_app.get(img_bgr)
    f = largest_face(faces)
    if f is None:
        return None
    return f.embedding.astype(np.float32)

def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def cosine_similarity(a, b):
    a = normalize(a)
    b = normalize(b)
    return float(np.dot(a, b))
