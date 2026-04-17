import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.linalg import norm
import json
import os

class TravelEnv(gym.Env):
    """
    Môi trường Học tăng cường cho bài toán Gợi ý Lộ trình Du lịch
    """
    def __init__(self, places_data=None, matrix_data=None, embedding_dim=1536):
        super(TravelEnv, self).__init__()
        
        # 1. Nạp dữ liệu (Load Data)
        if places_data is not None and matrix_data is not None:
            self.places = places_data
            self.time_matrix = matrix_data
        else:
            # Nếu chưa có file thật, dùng hàm tạo dữ liệu giả để test
            print("⚠️ CẢNH BÁO: Đang sử dụng dữ liệu giả lập (Mock Data) để kiểm thử.")
            self.places, self.time_matrix = self._generate_mock_data(embedding_dim)
            
        self.num_places = len(self.places)
        self.embedding_dim = embedding_dim
        
        # 2. Định nghĩa Không gian Hành động (Action Space)
        # Action từ 0 đến (N-1): Chọn đi đến POI tương ứng.
        # Action N: Hành động "Kết thúc chuyến đi sớm" (Về khách sạn).
        self.action_space = spaces.Discrete(self.num_places + 1)
        
        # 3. Định nghĩa Không gian Trạng thái (Observation Space)
        self.observation_space = spaces.Dict({
            "current_time": spaces.Box(low=0, high=1440, shape=(1,), dtype=np.float32), # Phút trong ngày
            "current_location": spaces.Discrete(self.num_places),
            "visited_mask": spaces.MultiBinary(self.num_places) # 1 là đã đi, 0 là chưa
        })
        
        # Biến toàn cục cho mỗi Episode
        self.user_embedding = None
        self.current_time = 0.0
        self.current_location = 0
        self.visited = None

    def _cosine_sim(self, user_vec, place_vec):
        """Tính độ tương đồng Vibe (Cosine Similarity)"""
        u, p = np.array(user_vec), np.array(place_vec)
        if norm(u) == 0 or norm(p) == 0: return 0
        return np.dot(u, p) / (norm(u) * norm(p))

    def reset(self, seed=None, options=None):
        """Khởi tạo lại môi trường (Bắt đầu 1 ngày mới)"""
        super().reset(seed=seed)
        
        # Lấy User Embedding từ hệ thống, nếu không có thì random
        if options and 'user_embedding' in options:
            self.user_embedding = options['user_embedding']
        else:
            self.user_embedding = np.random.rand(self.embedding_dim)
            
        self.current_time = 480.0  # Bắt đầu lúc 08:00 AM (8 * 60)
        self.current_location = 0  # Giả sử Index 0 luôn là Khách sạn/Điểm xuất phát
        self.visited = np.zeros(self.num_places, dtype=np.int8)
        self.visited[0] = 1 # Đánh dấu đã ở KS
        
        return self._get_obs(), {}

    def _get_obs(self):
        """Đóng gói trạng thái hiện tại"""
        return {
            "current_time": np.array([self.current_time], dtype=np.float32),
            "current_location": self.current_location,
            "visited_mask": self.visited.copy()
        }

    def step(self, action):
        """Agent thực hiện 1 hành động (chọn 1 địa điểm)"""
        info = {"poi_name": "Kết thúc chuyến đi"}
        
        # 1. Xử lý hành động: KẾT THÚC
        if action == self.num_places:
            return self._get_obs(), 0.0, True, False, info
            
        info["poi_name"] = self.places[action]['name']
        
        # 2. Ràng buộc cứng: ĐÃ ĐI RỒI
        if self.visited[action] == 1:
            return self._get_obs(), -100.0, True, False, {"reason": "Lỗi: Đi lại điểm cũ", **info}

        # Lấy thông số di chuyển & tham quan
        travel_time = self.time_matrix[self.current_location][action]
        
        # --- FIX LỖI MISSING FIELD Ở ĐÂY ---
        # Dùng .get() để lấy giá trị, nếu trong DB bị thiếu thì gán mặc định là 60 phút
        stay_time = self.places[action].get('avg_stay_minutes', 60)
        arrival_time = self.current_time + travel_time
        
        # 3. Ràng buộc cứng: GIỜ MỞ CỬA
        # Tương tự, nếu DB thiếu activity_hours, mặc định mở cửa 24/24 [0, 1440]
        open_t, close_t = self.places[action].get('activity_hours', [0, 1440])
        
        if arrival_time < open_t or (arrival_time + stay_time) > close_t:
            return self._get_obs(), -50.0, True, False, {"reason": "Lỗi: Đóng cửa/Quá giờ", **info}

        # --- NẾU HỢP LỆ, CẬP NHẬT TRẠNG THÁI ---
        self.current_time += travel_time + stay_time
        self.current_location = action
        self.visited[action] = 1
        
        # --- TÍNH TOÁN REWARD ---
        # Điểm Vibe (0.0 -> 1.0)
        vibe_score = self._cosine_sim(self.user_embedding, self.places[action]['embedding'])
        
        # Công thức Reward (Quan trọng nhất)
        # Thưởng lớn nếu hợp Vibe, Phạt nhẹ dựa trên thời gian di chuyển
        reward = (vibe_score * 50.0) - (travel_time * 0.2) 
        
        # 4. Kiểm tra điều kiện dừng (Hết ngày)
        terminated = False
        if self.current_time >= 1320.0: # 22:00 PM (22 * 60)
            terminated = True
            info["reason"] = "Hoàn thành ngày"
            
        return self._get_obs(), reward, terminated, False, info

    def _generate_mock_data(self, dim, num_places=20):
        """Hàm phụ trợ sinh dữ liệu giả để test môi trường"""
        places = []
        # Điểm 0 là Khách sạn
        places.append({
            "poi_id": 0, "name": "Khách sạn Xuất phát",
            "activity_hours": [0, 1440], "avg_stay_minutes": 0,
            "embedding": np.random.rand(dim).tolist()
        })
        # Sinh 19 điểm đến ngẫu nhiên
        for i in range(1, num_places):
            places.append({
                "poi_id": i, "name": f"Địa điểm du lịch {i}",
                "activity_hours": [480, 1260], # 8h sáng - 21h tối
                "avg_stay_minutes": np.random.choice([60, 90, 120]),
                "embedding": np.random.rand(dim).tolist()
            })
            
        # Sinh ma trận thời gian giả (từ 5 đến 45 phút)
        time_matrix = np.random.randint(5, 45, size=(num_places, num_places))
        np.fill_diagonal(time_matrix, 0) # Khoảng cách từ A đến A = 0
        
        return places, time_matrix

# ==========================================
# KHỐI TEST NHANH (SANITY CHECK)
# ==========================================
if __name__ == "__main__":
    print("--- KHỞI ĐỘNG BÀI KIỂM TRA MÔI TRƯỜNG ---")
    
    places_file = 'placedata_19042026.json'
    matrix_file = 'travel_time_matrix.npy'
    
    # 1. THỬ NẠP DỮ LIỆU THẬT
    if os.path.exists(places_file) and os.path.exists(matrix_file):
        print(f"Đang nạp dữ liệu thật từ '{places_file}' và '{matrix_file}'...")
        with open(places_file, 'r', encoding='utf-8') as f:
            real_places = json.load(f)
        real_matrix = np.load(matrix_file)
        
        # Tự động nhận diện số chiều embedding từ data thật
        actual_dim = len(real_places[0]['embedding'])
        env = TravelEnv(places_data=real_places, matrix_data=real_matrix, embedding_dim=actual_dim)
        print(f"✅ Nạp thành công {env.num_places} địa điểm với Embedding {actual_dim} chiều!")
    else:
        # 2. DÙNG MOCK DATA NẾU CHƯA CÓ FILE
        print(f"⚠️ Không tìm thấy file data thật. Đang dùng dữ liệu giả lập (Mock Data).")
        print("👉 Mẹo: Hãy chắc chắn bạn đã chạy script tạo data và để file cùng thư mục.")
        env = TravelEnv(embedding_dim=10)
    
    # Tạo một User Vector giả định (cùng số chiều với môi trường)
    dummy_user = np.random.rand(env.embedding_dim)
    
    # Bắt đầu ngày mới
    obs, _ = env.reset(options={"user_embedding": dummy_user})
    print(f"Trạng thái ban đầu: Thời gian={obs['current_time'][0]} (8:00 AM), Vị trí={obs['current_location']}")
    
    done = False
    total_reward = 0
    step_count = 1
    
    # Agent chạy NGẪU NHIÊN để test Môi trường
    while not done:
        # Chọn ngẫu nhiên 1 hành động hợp lệ (từ 1 đến num_places, bỏ KS số 0)
        action = np.random.randint(1, env.num_places + 1) 
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        time_format = f"{int(obs['current_time'][0]//60):02d}:{int(obs['current_time'][0]%60):02d}"
        
        if reward < 0 and "Lỗi" in info.get("reason", ""):
            print(f"Bước {step_count}: ❌ Agent đi sai luật -> Chọn '{info['poi_name']}' - Lỗi: {info['reason']} (Reward: {reward:.2f})")
        else:
            print(f"Bước {step_count}: ✅ Agent đi đến '{info['poi_name']}' -> Thời gian hiện tại: {time_format} (Reward: {reward:.2f})")
            
        step_count += 1
        
    print("-" * 30)
    print(f"Kết thúc Episode! Tổng điểm (Total Reward): {total_reward:.2f}")
    print("Trạng thái Môi trường: HOẠT ĐỘNG TỐT \u2728")