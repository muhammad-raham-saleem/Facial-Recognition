# Facial Recognition

Short FastAPI + webcam demo using InsightFace embeddings.

## Run

1) Install deps (example):

```bash
pip install fastapi uvicorn insightface opencv-python requests numpy
```

2) Start API:

```bash
uvicorn backend.main:app --reload
```

3) Enroll a user (3+ images):

```bash
curl -F "images=@/path/to/img1.jpg" \
     -F "images=@/path/to/img2.jpg" \
     -F "images=@/path/to/img3.jpg" \
     "http://localhost:8000/enroll?user_id=alice"
```

4) Verify from webcam:

- Edit `backend/camera_verify.py` and set `USER_ID = "alice"`.
- Run:

```bash
python backend/camera_verify.py
```

Embeddings are saved to `backend/data/embeddings/<user_id>.npy`.

