# person_ReID_tracking_software
Cross-video person re-identification using YOLO11 and Deep SORT with OSNet. Assigns consistent global IDs, captures thumbnails, and matches using cosine and histogram similarity. Includes visual tracking, colored ID boxes, and similarity analytics.

# Cross-Video Person Re-Identification with YOLO11 and Deep SORT

This project implements a robust person tracking and re-identification system across multiple videos using:
- **YOLO11** for real-time object detection
- **Deep SORT** with **OSNet** for person re-identification

The system maintains consistent **Global IDs (GIDs)** across different frames and videos by combining **cosine similarity** of feature embeddings with **color histogram similarity**.

---

## 🚀 Features

- **Real-time person detection** using YOLO11  
- **Accurate ID assignment** via Deep SORT with OSNet embeddings  
- **Cross-video tracking** with a persistent global memory  
- **Angle-aware updates**: captures thumbnails every 30 frames to adapt to changes in pose and appearance  
- **Cosine + HSV histogram matching** to boost re-ID reliability  
- **Dynamic bounding box colors** per person for clearer visuals  
- **Similarity analysis tools**: Cosine & Histogram similarity reports and histograms  


---

## 📊 Sample Outputs

- **Tracked Video:** Bounding boxes with consistent GIDs and distinct colors  
- **CSV Reports:** Global tracking, similarity stats, and unknown suggestions  
- **Similarity Histograms:** Visualize cosine and histogram trends per GID  

---

## 🛠 Requirements

```bash
pip install -r requirements.txt
