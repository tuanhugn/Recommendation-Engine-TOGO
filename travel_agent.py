import numpy as np
import json
from travel_env import TravelEnv

class TravelAgent:
    """
    Bộ não AI điều khiển quá trình tìm kiếm lộ trình dựa trên Môi trường.
    """
    def __init__(self, env: TravelEnv):
        self.env = env

    def get_best_action(self, obs):
        """
        Thuật toán Tìm kiếm (Mini-MCTS / Greedy Lookahead):
        Nhìn trước 1 bước toàn bộ 600+ địa điểm để chọn nước đi tối ưu nhất.
        """
        best_action = -1
        best_reward = -float('inf')
        
        current_loc = obs['current_location']
        current_time = obs['current_time'][0]
        visited = obs['visited_mask']
        
        # Duyệt qua tất cả các điểm đến để giả lập (Lookahead)
        for action in range(self.env.num_places):
            if visited[action] == 1:
                continue # Bỏ qua điểm đã đi
                
            # Lấy thông số từ Môi trường
            travel_time = self.env.time_matrix[current_loc][action]
            stay_time = self.env.places[action].get('avg_stay_minutes', 60)
            arrival_time = current_time + travel_time
            open_t, close_t = self.env.places[action].get('activity_hours', [0, 1440])
            
            # Ràng buộc: Đóng cửa thì không tới
            if arrival_time < open_t or (arrival_time + stay_time) > close_t:
                continue 
                
            # Dữ liệu đã được ép kiểu sạch sẽ ở khối __main__ bên dưới
            place_emb = self.env.places[action].get('embedding')
            if place_emb is None:
                place_emb = np.zeros_like(self.env.user_embedding)
                
            # Tính toán phần thưởng giả lập (Reward)
            try:
                vibe_score = self.env._cosine_sim(self.env.user_embedding, place_emb)
            except Exception:
                vibe_score = 0.0 # Bỏ qua nếu có lỗi rác toán học không lường trước
                
            reward = (vibe_score * 50.0) - (travel_time * 0.2)
            
            # Chọn hành động có điểm thưởng cao nhất
            if reward > best_reward:
                best_reward = reward
                best_action = action
                
        # Nếu không còn điểm nào thỏa mãn (hết giờ, hoặc hết điểm đi), chọn hành động Kết Thúc
        if best_action == -1:
            return self.env.num_places
            
        return best_action

    def plan_multi_day_trip(self, user_embedding, days=3):
        """
        Lên lịch trình cho nhiều ngày liên tiếp.
        """
        print(f"🤖 AI đang tính toán lộ trình {days} ngày...")
        global_visited = None
        full_itinerary = []
        total_trip_reward = 0
        
        for day in range(1, days + 1):
            # 1. Bắt đầu ngày mới trong Môi trường
            obs, _ = self.env.reset(options={"user_embedding": user_embedding})
            
            # 2. Ghi đè trí nhớ (Những điểm đã đi từ các ngày trước)
            if global_visited is not None:
                self.env.visited = global_visited.copy()
                obs['visited_mask'] = global_visited.copy()
                
            done = False
            day_plan = []
            
            # 3. Agent liên tục chọn nước đi cho đến khi hết ngày
            while not done:
                action = self.get_best_action(obs)
                obs, reward, done, truncated, info = self.env.step(action)
                
                # Nếu không phải hành động Kết thúc, thêm vào lịch trình
                if action != self.env.num_places and reward > 0:
                    total_trip_reward += reward
                    
                    # Tính giờ đến và giờ đi
                    time_leave = obs['current_time'][0]
                    stay_time = self.env.places[action].get('avg_stay_minutes', 60)
                    time_arrive = time_leave - stay_time
                    
                    time_format = f"{int(time_arrive//60):02d}:{int(time_arrive%60):02d} - {int(time_leave//60):02d}:{int(time_leave%60):02d}"
                    
                    day_plan.append({
                        "time": time_format,
                        "place": info['poi_name'],
                        "vibe_reward": round(reward, 2)
                    })
                    
            # 4. Lưu lại trí nhớ cho ngày hôm sau
            global_visited = self.env.visited.copy()
            full_itinerary.append({"day": day, "activities": day_plan})
            
        return full_itinerary, total_trip_reward

# ==========================================
# KHỐI TEST BỘ NÃO AI
# ==========================================
if __name__ == "__main__":
    import os
    
    places_file = 'placedata_19042026.json'
    matrix_file = 'travel_time_matrix.npy'
    
    if os.path.exists(places_file) and os.path.exists(matrix_file):
        with open(places_file, 'r', encoding='utf-8') as f:
            real_places = json.load(f)
        real_matrix = np.load(matrix_file)
        
        # --- FIX LỖI ĐỌC SỐ CHIỀU ---
        # Kiểm tra trước cấu trúc để đọc số chiều cho đúng (phòng bị string array)
        sample_emb = real_places[0].get('embedding', [])
        if isinstance(sample_emb, str):
            try:
                sample_emb = json.loads(sample_emb)
            except:
                sample_emb = [0]*768 # Fallback
        
        actual_dim = len(sample_emb) if len(sample_emb) > 0 else 768
        
        # --- FIX TRỊỆT ĐỂ: ÉP KIỂU STRING SANG LIST CHO TOÀN BỘ DATA ---
        for place in real_places:
            raw_emb = place.get('embedding')
            if isinstance(raw_emb, str):
                try:
                    place['embedding'] = json.loads(raw_emb)
                except Exception:
                    place['embedding'] = [0.0] * actual_dim
            elif raw_emb is None:
                place['embedding'] = [0.0] * actual_dim
        
        # 1. Khởi tạo Môi trường với dữ liệu đã được làm sạch
        env = TravelEnv(places_data=real_places, matrix_data=real_matrix, embedding_dim=actual_dim)
        
        # 2. Khởi tạo Agent và kết nối vào Môi trường
        agent = TravelAgent(env)
        
        # 3. Chạy Demo
        dummy_user = np.random.rand(actual_dim) # Thay bằng Embedding lấy từ Câu Prompt của User
        
        itinerary, total_score = agent.plan_multi_day_trip(user_embedding=dummy_user, days=3)
        
        print("\n" + "="*40)
        print(f"🎉 LỘ TRÌNH ĐỀ XUẤT (TỔNG ĐIỂM: {total_score:.2f})")
        print("="*40)
        
        for day_plan in itinerary:
            print(f"\n☀️ NGÀY {day_plan['day']}:")
            for act in day_plan['activities']:
                print(f"  [{act['time']}] {act['place']} (Vibe: {act['vibe_reward']})")
    else:
        print("Vui lòng chạy file với dữ liệu thật để xem lịch trình!")