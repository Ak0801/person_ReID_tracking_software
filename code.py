import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.metrics.pairwise import cosine_similarity
import random

# PATHS
VIDEO_FOLDER = 'input_videos'
OUTPUT_FOLDER = 'output'
CSV_FILE = os.path.join(OUTPUT_FOLDER, 'csv_files/global_id_tracking.csv')
SIMILARITY_REPORT = os.path.join(OUTPUT_FOLDER, 'csv_files/similarity_report.csv')

# Load YOLO11 model
yolo_model = YOLO('yolo11m.pt')

# Load DeepSort with ReID
deep_sort = DeepSort(
    max_age=30,
    nn_budget=100,
    override_track_class=None,
    embedder="torchreid",
    embedder_model_name="osnet_x1_0",
    embedder_wts="models\osnet_x1_0_msmt17.pth",
    half=True,
    bgr=True,
)

# Global memory and utilities
global_memory = {}
rows = []
similarity_logs = []
gid_stats = []
unk_stats = []

# Configs
SIMILARITY_THRESHOLD = 0.70
COS_MIN = 0.60
HIST_MIN = 0.70
VIDEO2_COSINE_THRESHOLD = 0.60
VIDEO2_HIST_THRESHOLD = 0.60
MAX_EMB_HISTORY = 10


color_map = {}

def get_color(gid):
    if gid not in color_map:
        color_map[gid] = tuple(random.randint(0, 255) for _ in range(3))
    return color_map[gid]



def extract_histogram(img):
    if img.size == 0:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def histogram_similarity(h1, h2):
    return cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CORREL)

def suggest_best_gid_for_unk(unk_sim, global_stats):
    best_gid = None
    best_cos_sim = 0
    best_hist_sim = 0
    for gid, stats in global_stats.items():
        if stats['cos_sim'] > best_cos_sim and stats['hist_sim'] > best_hist_sim:
            best_cos_sim = stats['cos_sim']
            best_hist_sim = stats['hist_sim']
            best_gid = gid
    return best_gid

global_gid_stats = {}

unk_counter = 0

# Process videos
for video_index, video_file in enumerate(sorted(os.listdir(VIDEO_FOLDER))):
    if not video_file.endswith('.mp4'):
        continue

    video_path = os.path.join(VIDEO_FOLDER, video_file)
    cap = cv2.VideoCapture(video_path)

    out_path = os.path.join(OUTPUT_FOLDER, f'{os.path.splitext(video_file)[0]}_output.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f'Processing {video_file}')
    used_gids_in_frame = set()
    local_track_history = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, conf=0.8)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        person_dets = []
        for i, cls_id in enumerate(class_ids):
            if cls_id == 0:
                x1, y1, x2, y2 = boxes[i]
                w, h = x2 - x1, y2 - y1
                person_dets.append(([x1, y1, w, h], confs[i], 'person'))

        tracks = deep_sort.update_tracks(person_dets, frame=frame)
        assigned_gids_this_frame = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            local_id = track.track_id
            if local_id not in local_track_history:
                local_track_history[local_id] = 1
                continue
            else:
                local_track_history[local_id] += 1

            if local_track_history[local_id] < 20:
                continue

            x1, y1, x2, y2 = map(int, track.to_tlbr())
            if y2 <= y1 or x2 <= x1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or len(track.features) == 0:
                continue

            query_emb = np.mean(track.features[-MAX_EMB_HISTORY:], axis=0)
            hist_feat = extract_histogram(crop)

            best_score = -1
            best_gid = None
            best_cosine = 0
            best_hist = 0
            match_report = []

            for gid, mem in global_memory.items():
                if video_index > 0 and str(gid).startswith("unk"):
                    continue

                if gid in assigned_gids_this_frame:
                    continue
                sim = cosine_similarity([query_emb], [mem['centroid']])[0][0]
                hist_sims = [histogram_similarity(hist_feat, h) for h in mem['histograms'] if h is not None]
                max_hist = max(hist_sims) if hist_sims else 0
                match_report.append([video_file, frame_idx, local_id, gid, sim, max_hist, 0])

                if video_index == 0:
                    if sim > SIMILARITY_THRESHOLD or (sim > COS_MIN and max_hist > HIST_MIN):
                        if sim > best_score:
                            best_score = sim
                            best_gid = gid
                            best_cosine = sim
                            best_hist = max_hist
                else:
                    # Match if either cosine OR histogram similarity is above threshold
                    if sim > VIDEO2_COSINE_THRESHOLD or max_hist > VIDEO2_HIST_THRESHOLD:
                        combined_score = sim + max_hist  
                        if combined_score > best_score:
                            best_score = combined_score
                            best_gid = gid
                            best_cosine = sim
                            best_hist = max_hist


            if video_index == 0:
                if best_gid is not None:
                    gid = best_gid
                    global_memory[gid]['embeddings'].append(query_emb)
                    global_memory[gid]['centroid'] = np.mean(global_memory[gid]['embeddings'], axis=0)
                    global_memory[gid]['histograms'].append(hist_feat)
                    gid_stats.append([gid, best_cosine, best_hist, 0])
                else:
                    gid = len(global_memory)
                    global_memory[gid] = {
                        'embeddings': [query_emb],
                        'centroid': query_emb,
                        'histograms': [hist_feat]
                    }
                    
                    gid_stats.append([gid, 1.0, 1.0, 0])
            else:
                gid = best_gid
                if gid is None:
                    unk_label = f'unk{unk_counter}'
                    suggested_gid = suggest_best_gid_for_unk(
                        {'cos_sim': best_cosine, 'hist_sim': best_hist}, global_gid_stats
                    )
                    unk_stats.append([unk_label, video_file, frame_idx, best_cosine, best_hist, suggested_gid])
                    gid = unk_label
                global_memory[gid] = {
                    'embeddings': [query_emb],
                    'centroid': query_emb,
                    'histograms': [hist_feat]
                }
                unk_counter += 1

            assigned_gids_this_frame.add(gid)
            match_report = [[gid] + r for r in match_report]
            similarity_logs.extend(match_report)

            cv2.rectangle(frame, (x1, y1), (x2, y2), get_color(gid), 2)
            
                        
            

            display_label = f'{gid}' if str(gid).startswith('unk') else f'ID:{gid}'

            cv2.putText(frame, display_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Update thumbnails and features every 30 frames
            if frame_idx % 30 == 0:
                
                global_memory[gid]['embeddings'].append(query_emb)
                global_memory[gid]['histograms'].append(hist_feat)
                global_memory[gid]['centroid'] = np.mean(global_memory[gid]['embeddings'], axis=0)

            rows.append([gid, video_file, frame_idx, x1, y1, x2 - x1, y2 - y1])

        out_writer.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out_writer.release()


pd.DataFrame(rows, columns=[
    'global_id', 'video', 'frame', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']
).to_csv(CSV_FILE, index=False)


pd.DataFrame(similarity_logs, columns=[
    'assigned_gid', 'video', 'frame', 'local_track_id', 'compared_to_gid', 'cosine_sim', 'hist_sim', 'iou']
).to_csv(SIMILARITY_REPORT, index=False)


print(f" Global tracking saved to {CSV_FILE}")
print(f" Similarity report saved to {SIMILARITY_REPORT}")



