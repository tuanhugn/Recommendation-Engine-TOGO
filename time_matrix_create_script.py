import json
import math
import numpy as np
import os

def haversine(lat1, lon1, lat2, lon2):
    """Tính khoảng cách đường chim bay giữa 2 tọa độ (km)"""
    R = 6371.0 # Bán kính trái đất (km)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2.0)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def build_matrix():
    input_file = 'placedata_19042026.json'
    output_file = 'travel_time_matrix.npy'
    
    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file {input_file}. Hãy export từ DB ra trước!")
        return

    print("Đang đọc dữ liệu địa điểm...")
    with open(input_file, 'r', encoding='utf-8') as f:
        places = json.load(f)
        
    num_places = len(places)
    print(f"Tổng số địa điểm: {num_places}")
    
    # Khởi tạo ma trận toàn số 0
    time_matrix = np.zeros((num_places, num_places), dtype=np.float32)
    
    print("Đang tính toán ma trận thời gian (Haversine + Traffic Delay)...")
    for i in range(num_places):
        for j in range(num_places):
            if i == j:
                time_matrix[i][j] = 0.0 # Ở nguyên một chỗ tốn 0 phút
            else:
                dist_km = haversine(
                    places[i]['latitude'], places[i]['longitude'], 
                    places[j]['latitude'], places[j]['longitude']
                )
                # Giả định: Tốc độ trung bình nội thành là 20km/h
                # Thời gian = (Quãng đường / Tốc độ) * 60 phút + 10 phút hao phí (gửi xe, đi bộ)
                travel_time_minutes = (dist_km / 20.0) * 60.0 + 10.0
                time_matrix[i][j] = round(travel_time_minutes)
                
    # Lưu ra file nhị phân numpy (npy) để load siêu tốc độ
    np.save(output_file, time_matrix)
    print(f"✅ Đã tạo thành công {output_file} (Kích thước: {num_places}x{num_places})")

if __name__ == "__main__":
    build_matrix()