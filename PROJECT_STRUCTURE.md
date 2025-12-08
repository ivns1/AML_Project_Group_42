# AML Project - Feathers in Focus

## Proje Genel Bakis

Bu proje, **UvA (Universiteit van Amsterdam) Applied Machine Learning** dersi icin bir grup projesidir. Kaggle yarisma formatinda, kus turlerini siniflandirma gorevidir.

### Proje Bilgileri
| Ozellik | Deger |
|---------|-------|
| **Ders** | UvA Applied Machine Learning |
| **Proje Adi** | Feathers in Focus |
| **Gorev** | Kus Turu Siniflandirma (Image Classification) |
| **Sinif Sayisi** | 200 farkli kus turu |
| **Kaggle Link** | https://www.kaggle.com/t/0e9856f5cb5f40af8739be017cc75b9b |

---

## Veri Seti Ozeti

### Istatistikler
| Metrik | Deger |
|--------|-------|
| **Egitim Goruntuleri** | 3,926 adet |
| **Test Goruntuleri** | 4,000 adet |
| **Toplam Sinif** | 200 kus turu |
| **Attribute Sayisi** | 312 ozellik |
| **Egitim Veri Boyutu** | ~369 MB |
| **Test Veri Boyutu** | ~379 MB |

### Sinif Basina Ortalama Goruntu
- Egitim: ~19.6 goruntu/sinif
- Test: 20 goruntu/sinif

---

## Dizin Yapisi

```
AML_Project/
├── aml-2025-feathers-in-focus/     # Ana veri klasoru
│   ├── train_images/               # Egitim goruntuleri
│   │   └── train_images/
│   │       └── *.jpg               # 3,926 JPG dosyasi
│   ├── test_images/                # Test goruntuleri
│   │   └── test_images/
│   │       └── *.jpg               # 4,000 JPG dosyasi
│   ├── train_images.csv            # Egitim etiketleri
│   ├── test_images_sample.csv      # Submission ornegi
│   ├── test_images_path.csv        # Test dosya yollari
│   ├── class_names.npy             # Sinif isimleri (numpy)
│   ├── attributes.npy              # Attribute matrisi (numpy)
│   └── attributes.txt              # Attribute isimleri (text)
│
├── class_names.csv                 # Islenmis sinif isimleri
├── attribute_names.csv             # Islenmis attribute isimleri
├── attributes.csv                  # Islenmis attribute matrisi
├── Preprocessing.py                # Veri on-isleme scripti
├── Group_project_description.pdf   # Proje aciklamasi
└── README.md                       # Temel bilgiler
```

---

## Dosya Formatlari ve Icerikleri

### 1. train_images.csv
Egitim verisi etiketlerini icerir.

| Sutun | Aciklama |
|-------|----------|
| `image_path` | Goruntu dosya yolu (ornek: `/train_images/1.jpg`) |
| `label` | Sinif etiketi (1-200 arasi) |

**Ornek:**
```csv
image_path,label
/train_images/1.jpg,1
/train_images/2.jpg,1
/train_images/3.jpg,1
```

### 2. test_images_sample.csv
Kaggle submission formati. **Sonuc dosyasi bu formatta olmali!**

| Sutun | Aciklama |
|-------|----------|
| `id` | Test ornegi ID'si (1-4000) |
| `label` | Tahmin edilen sinif (1-200) |

**Ornek:**
```csv
id,label
1,1
2,1
3,1
```

### 3. class_names.csv
200 kus turunun isimleri.

| Sutun | Aciklama |
|-------|----------|
| `Label` | Sinif numarasi (1-200) |
| `Class_name` | Kus turu ismi |

**Ornek Siniflar:**
| Label | Class_name |
|-------|------------|
| 1 | Black_footed_Albatross |
| 2 | Laysan_Albatross |
| 17 | Cardinal |
| 200 | Common_Yellowthroat |

### 4. attribute_names.csv
312 farkli ozellik (attribute) tanimlari.

| Sutun | Aciklama |
|-------|----------|
| `id` | Attribute ID (1-312) |
| `attribute_group` | Ozellik grubu |
| `attribute` | Ozellik degeri |

**Attribute Gruplari:**
- `has_bill_shape`: Gaga sekli (curved, dagger, hooked, needle, cone, vb.)
- `has_wing_color`: Kanat rengi (blue, brown, grey, yellow, vb.)
- `has_upperparts_color`: Ust kisim rengi
- `has_underparts_color`: Alt kisim rengi
- `has_breast_pattern`: Gogus deseni
- `has_back_color`: Sirt rengi
- `has_tail_shape`: Kuyruk sekli
- `has_upper_tail_color`: Ust kuyruk rengi
- `has_head_pattern`: Bas deseni
- `has_breast_color`: Gogus rengi
- `has_throat_color`: Bogaz rengi
- `has_eye_color`: Goz rengi
- `has_bill_length`: Gaga uzunlugu
- `has_forehead_color`: Alin rengi
- `has_under_tail_color`: Alt kuyruk rengi
- `has_nape_color`: Ense rengi
- `has_belly_color`: Karin rengi
- `has_wing_shape`: Kanat sekli
- `has_size`: Boyut
- `has_shape`: Genel sekil
- `has_back_pattern`: Sirt deseni
- `has_tail_pattern`: Kuyruk deseni
- `has_belly_pattern`: Karin deseni
- `has_primary_color`: Ana renk
- `has_leg_color`: Bacak rengi
- `has_bill_color`: Gaga rengi
- `has_crown_color`: Tepe rengi
- `has_wing_pattern`: Kanat deseni

### 5. attributes.csv
Her sinif icin 312 attribute'un olasilik degerleri (0.0 - 1.0 arasi).

| Sutun | Aciklama |
|-------|----------|
| `Class_id` | Sinif numarasi (1-200) |
| `1-312` | Her attribute icin olasilik skoru |

**Boyut:** 200 satir x 313 sutun (Class_id + 312 attribute)

---

## Preprocessing.py Scripti

Bu script, numpy formatindaki ham verileri CSV formatina donusturur:

### Islevler:
1. **class_names.npy -> class_names.csv**
   - Sinif isimlerini okunabilir CSV formatina cevirir

2. **attributes.txt -> attribute_names.csv**
   - Attribute tanimlarini ID, grup ve isim olarak ayirir

3. **attributes.npy -> attributes.csv**
   - Her sinif icin attribute matrisini CSV'ye cevirir

### Kullanim:
```bash
python Preprocessing.py
```

---

## Proje Gereksinimleri

### Gorevler:
1. **Baseline Model**: HuggingFace'den pretrained model kullanarak baseline olusturmak
   - Kaggle'a "baseline" adi ile submit etmek

2. **Custom Model**: Kendi ML modelini gelistirmek ve baseline'i gecmeye calismak

3. **Analiz**:
   - Computational complexity karsilastirmasi
   - Error analysis (modelin nerede hatali tahmin yaptigi)

### Ciktilar:
- **Poster**: Ana bulgular, model mimarisi, sonuclar
- **Kaggle Submission**: test_images_sample.csv formatinda

### Poster Icerigi:
- Ana arastirma sorusu
- Model mimarisi figuru
- Baseline karsilastirmasi
- Hesaplama karmasikligi analizi
- Hata analizi (1-2 ornek)

---

## Onerilen Yaklasimlar

### 1. Baseline (HuggingFace Pretrained)
- ResNet, EfficientNet, ViT gibi pretrained modeller
- Fine-tuning ile kus siniflandirmaya adapte etme

### 2. Custom Model Fikirleri
- CNN tabanli mimari (daha az parametre ile)
- Attribute bilgisini kullanma (multi-task learning)
- Transfer learning ile kucuk model
- Data augmentation teknikleri

### 3. Attribute Kullanimi
Attributes.csv dosyasi ek bilgi saglar:
- Her sinifin gorsel ozellikleri tanimli
- Bu bilgi auxiliary loss olarak kullanilabilir
- Zero-shot veya few-shot ogrenme icin faydali olabilir

---

## Hizli Baslangic

### 1. Veri Yukleme (PyTorch)
```python
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class BirdDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.root_dir + self.data.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['label'] - 1  # 0-indexed

        if self.transform:
            image = self.transform(image)

        return image, label
```

### 2. Sinif Isimlerini Yukleme
```python
import pandas as pd

class_names = pd.read_csv('class_names.csv')
label_to_name = dict(zip(class_names['Label'], class_names['Class_name']))

# Ornek: label_to_name[1] = 'Black_footed_Albatross'
```

### 3. Attribute Bilgisini Yukleme
```python
import pandas as pd

attributes = pd.read_csv('attributes.csv')
attr_names = pd.read_csv('attribute_names.csv')

# Sinif 1'in attribute vektoru
class_1_attrs = attributes[attributes['Class_id'] == 1].iloc[0, 1:].values
```

---

## Onemli Notlar

1. **Submission Formati**: test_images_sample.csv ile ayni formatta olmali
2. **Label Aralik**: 1-200 (0-199 degil!)
3. **Baseline Isimlendirme**: Kaggle'a submit ederken "baseline" olarak adlandirmak
4. **Computational Cost**: Model performansi kadar hesaplama maliyeti de onemli

---

## Dosya Yollari Referansi

| Dosya | Yol |
|-------|-----|
| Egitim CSV | `aml-2025-feathers-in-focus/train_images.csv` |
| Test Sample | `aml-2025-feathers-in-focus/test_images_sample.csv` |
| Egitim Goruntuleri | `aml-2025-feathers-in-focus/train_images/train_images/*.jpg` |
| Test Goruntuleri | `aml-2025-feathers-in-focus/test_images/test_images/*.jpg` |
| Sinif Isimleri | `class_names.csv` |
| Attribute Isimleri | `attribute_names.csv` |
| Attribute Matrisi | `attributes.csv` |

---

## Iletisim ve Kaynaklar

- **Kaggle Competition**: https://www.kaggle.com/t/0e9856f5cb5f40af8739be017cc75b9b
- **HuggingFace Models**: https://huggingface.co/models?pipeline_tag=image-classification
