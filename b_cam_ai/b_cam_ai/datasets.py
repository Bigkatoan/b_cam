import os
import sys
import cv2
import numpy as np
import torch
import random
import re
from torch.utils.data import Dataset, DataLoader

# --- [CRITICAL FIX 1] CHẶN XUNG ĐỘT LUỒNG OPENCV ---
# Bắt buộc OpenCV chạy đơn luồng để tránh tranh chấp CPU với PyTorch DataLoader
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# ---------------------------------------------------

# Thư viện chuẩn xử lý COCO
try:
    from pycocotools.coco import COCO
except ImportError:
    print("LỖI: Thiếu thư viện 'pycocotools'.")
    print("Vui lòng chạy: pip install pycocotools")
    sys.exit(1)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("LỖI: Thiếu thư viện 'albumentations'.")
    print("Vui lòng chạy: pip install albumentations")
    sys.exit(1)

# ==============================================================================
# 1. TỪ ĐIỂN CLASS
# ==============================================================================

COCO_CLASSES_VN = {
    1: 'người', 2: 'xe đạp', 3: 'ô tô', 4: 'xe máy', 5: 'máy bay', 6: 'xe buýt', 7: 'tàu hỏa', 8: 'xe tải', 9: 'thuyền',
    10: 'đèn giao thông', 11: 'vòi chữa cháy', 13: 'biển báo dừng', 14: 'đồng hồ đỗ xe', 15: 'ghế đá', 16: 'con chim',
    17: 'con mèo', 18: 'con chó', 19: 'con ngựa', 20: 'con cừu', 21: 'con bò', 22: 'con voi', 23: 'con gấu',
    24: 'ngựa vằn', 25: 'hươu cao cổ', 27: 'cái ba lô', 28: 'cái ô', 31: 'túi xách', 32: 'cà vạt', 33: 'vali',
    34: 'cái đĩa bay', 35: 'ván trượt', 36: 'ván tuyết', 37: 'bóng thể thao', 38: 'cái diều', 39: 'gậy bóng chày',
    40: 'găng tay bóng chày', 41: 'ván trượt', 42: 'tấm ván lướt', 43: 'vợt tennis', 44: 'cái chai', 46: 'ly rượu',
    47: 'cái cốc', 48: 'cái nĩa', 49: 'con dao', 50: 'cái thìa', 51: 'cái bát', 52: 'quả chuối', 53: 'quả táo',
    54: 'bánh sandwich', 55: 'quả cam', 56: 'bông cải xanh', 57: 'củ cà rốt', 58: 'xúc xích', 59: 'bánh pizza',
    60: 'bánh rán', 61: 'bánh kem', 62: 'cái ghế', 63: 'ghế sô pha', 64: 'chậu cây', 65: 'cái giường', 67: 'bàn ăn',
    70: 'nhà vệ sinh', 72: 'ti vi', 73: 'máy tính xách tay', 74: 'con chuột', 75: 'điều khiển từ xa', 76: 'bàn phím',
    77: 'điện thoại', 78: 'lò vi sóng', 79: 'lò nướng', 80: 'máy nướng bánh', 81: 'bồn rửa', 82: 'tủ lạnh',
    84: 'quyển sách', 85: 'đồng hồ', 86: 'lọ hoa', 87: 'cái kéo', 88: 'gấu bông', 89: 'máy sấy tóc', 90: 'bàn chải đánh răng'
}

# ==============================================================================
# 2. PROMPT GENERATOR (LOAD TỪ FILE)
# ==============================================================================

class VNPromptGenerator:
    """
    Load các mẫu câu từ file text (đã sinh sẵn bởi generate_templates.py hoặc Llama).
    """
    def __init__(self, template_file="data/vn_templates.txt"):
        self.templates = []
        
        # Tìm file template
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, template_file)
        
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # Lọc bỏ dòng trống
                self.templates = [line.strip() for line in lines if line.strip()]
            print(f"[PromptGen] Đã load {len(self.templates)} mẫu câu từ {file_path}")
        else:
            print(f"[PromptGen] CẢNH BÁO: Không tìm thấy {file_path}. Dùng mẫu mặc định.")
            self.templates = ["{obj}", "Hãy tìm {obj}"]

        self.templates_lower = [t.lower() for t in self.templates]

    def generate(self, class_names_list):
        if len(class_names_list) == 1:
            obj_str = class_names_list[0]
        else:
            joiner = random.choice([" và ", ", ", " cùng với ", " với ", " kèm "])
            obj_str = joiner.join(class_names_list)

        # 30% viết thường toàn bộ
        if random.random() < 0.3:
            template = random.choice(self.templates_lower)
        else:
            template = random.choice(self.templates)
            
        prompt = template.replace("{obj}", obj_str)
        prompt = re.sub(' +', ' ', prompt).strip()
        return prompt

# ==============================================================================
# 3. TOKENIZER
# ==============================================================================

class SimpleTokenizer:
    def __init__(self, max_length=20):
        self.max_length = max_length
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.vocab_size = 4
        
        all_words = set()
        
        # 1. Thêm từ vựng từ Class
        for name in COCO_CLASSES_VN.values():
            all_words.update(name.split())
            
        # 2. Thêm từ vựng từ File Template
        pg = VNPromptGenerator()
        all_templates = pg.templates + pg.templates_lower
        
        for sent in all_templates:
            clean_sent = sent.replace("{obj}", "")
            clean_sent = re.sub(r"[^a-zA-Z0-9\u00C0-\u1EF9 ]+", " ", clean_sent.lower())
            all_words.update(clean_sent.split())
                
        # 3. Thêm từ bổ trợ
        all_words.update(["và", "với", "cùng", "ở", "đâu", "gì", "nào", "của", "là", "trong", "ảnh"])
        
        for word in sorted(list(all_words)):
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.vocab_size += 1
                
        print(f"[Tokenizer] Vocab size (Tiếng Việt): {self.vocab_size}")

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\u00C0-\u1EF9 ]+", "", text)
        return text

    def encode(self, text):
        text = self._clean_text(text)
        tokens = text.split()
        ids = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in tokens]
        
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        else:
            ids = ids + [self.word2idx["<PAD>"]] * (self.max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)

# ==============================================================================
# 4. DATASET & HELPERS
# ==============================================================================

# --- [CRITICAL FIX 2] TỐI ƯU HÓA RAM CHO NOISE ---
def custom_light_noise(image, **kwargs):
    h, w, c = image.shape
    # Dùng float32 ngay từ đầu để giảm 50% RAM so với mặc định
    noise = np.random.randn(h, w, c).astype(np.float32) 
    factor = np.random.uniform(2.0, 8.0)
    
    # Ép kiểu an toàn, cộng trực tiếp (in-place) nếu có thể
    image_float = image.astype(np.float32)
    image_float += noise * factor
    
    # Clip và trả về uint8
    return np.clip(image_float, 0, 255).astype(np.uint8)
# --------------------------------------------------

class COCOVietPromptDataset(Dataset):
    def __init__(self, data_root, image_root, tokenizer=None, split='train', img_size=320):
        self.image_root = image_root
        self.split = split
        self.img_size = img_size
        
        ann_file = os.path.join(data_root, 'annotations', f'instances_{split}2014.json')
        if not os.path.exists(ann_file):
            ann_file = os.path.join(data_root, 'annotations', 'instances_trainval2014.json')
            
        sys.stdout = open(os.devnull, 'w')
        self.coco = COCO(ann_file)
        sys.stdout = sys.__stdout__
        
        self.img_ids = []
        all_ids = self.coco.getImgIds()
        for img_id in all_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                self.img_ids.append(img_id)
                
        # Prompt Generator load từ file
        self.prompt_gen = VNPromptGenerator()
        
        if tokenizer is None:
            self.tokenizer = SimpleTokenizer(max_length=20)
        else:
            self.tokenizer = tokenizer
            
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        if split == 'train':
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RGBShift(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.Lambda(name="CustomLightNoise", image=custom_light_noise, p=1.0),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ])
            
        print(f"[Dataset] Loaded {len(self.img_ids)} images ({split}).")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        try:
            img_id = self.img_ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            
            img_path = os.path.join(self.image_root, img_info['file_name'])
            image = cv2.imread(img_path)
            if image is None: return self.__getitem__(random.randint(0, len(self)-1))
            
            # Convert màu ngay lập tức để tiết kiệm logic sau này
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            present_cat_ids = list(set([ann['category_id'] for ann in anns]))
            
            if not present_cat_ids: return self.__getitem__((idx + 1) % len(self))
            
            num_targets = random.randint(1, min(2, len(present_cat_ids)))
            target_cat_ids = random.sample(present_cat_ids, k=num_targets)
            
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            for ann in anns:
                if ann['category_id'] in target_cat_ids:
                    m = self.coco.annToMask(ann)
                    mask = np.maximum(mask, m)
            
            target_names = [COCO_CLASSES_VN[cid] for cid in target_cat_ids]
            prompt_text = self.prompt_gen.generate(target_names)
            
            transformed = self.transform(image=image, mask=mask)
            image_tensor = transformed['image']
            mask_tensor = transformed['mask'].long()
            text_ids = self.tokenizer.encode(prompt_text)
            
            return {
                'image': image_tensor,
                'mask': mask_tensor,
                'text_ids': text_ids,
                'raw_prompt': prompt_text,
                'img_path': img_path
            }
        except Exception as e:
            # Fallback an toàn nếu lỗi
            return self.__getitem__((idx + 1) % len(self))

def get_dataloader(data_root, image_root, batch_size=32, img_size=320, split='train', num_workers=2):
    dataset = COCOVietPromptDataset(data_root, image_root, split=split, img_size=img_size)
    
    # --- [CRITICAL FIX 3] CẤU HÌNH LOADER AN TOÀN ---
    if num_workers > 0:
        # Giảm prefetch_factor xuống vừa đủ để không bị OOM
        prefetch_factor = 2 
        persistent_workers = True
    else:
        prefetch_factor = None
        persistent_workers = False
    
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(split=='train'),
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
        drop_last=(split=='train') # Bỏ batch cuối lẻ để ổn định BatchNorm
    )
    return loader, dataset.tokenizer.vocab_size

# Test & Debug
def save_debug_images(loader, output_dir="debug_samples", num_batches=1):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"\n[Debug] Đang lưu ảnh mẫu vào {output_dir}...")
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches: break
        images = batch['image']
        masks = batch['mask']
        prompts = batch['raw_prompt']
        for i in range(len(images)):
            img = images[i].permute(1, 2, 0).cpu().numpy()
            img = (img * std + mean) * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            mask = masks[i].cpu().numpy().astype(np.uint8)
            colored_mask = np.zeros_like(img); colored_mask[:, :, 1] = mask * 255 
            overlay = cv2.addWeighted(img, 0.6, colored_mask, 0.4, 0)
            filename = os.path.join(output_dir, f"batch{batch_idx}_img{i}.jpg")
            cv2.imwrite(filename, overlay)
            print(f"  Saved: {filename} | Prompt: \"{prompts[i]}\"")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.join(current_dir, "data", "coco") 
    IMG_ROOT = os.path.join(DATA_ROOT, "train2014")
    
    if os.path.exists(DATA_ROOT):
        # Kiểm tra xem file template đã có chưa
        template_file = os.path.join(current_dir, "data", "vn_templates.txt")
        if not os.path.exists(template_file):
            print("CHƯA THẤY FILE 'vn_templates.txt'.")
            print("Vui lòng chạy 'python generate_templates.py' trước!")
        else:
            print("File template OK. Testing loader...")
            loader, vocab = get_dataloader(DATA_ROOT, IMG_ROOT, batch_size=4, num_workers=0)
            save_debug_images(loader, num_batches=1)
    else:
        print("Không tìm thấy dữ liệu COCO.")